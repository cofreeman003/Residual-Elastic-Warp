"""
REwarp demo: stitch two arbitrary images using pretrained weights.

Usage:
    python demo.py --img1 reference.jpg --img2 target.jpg --out stitched.jpg
    python demo.py --img1 a.png --img2 b.png --out result.png --ckpt save/_TCell/epoch-last.pth
"""
import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import GaussianBlur

import utils
import models


def load_image(path):
    """Load image as a (1, 3, H, W) tensor in [-1, 1] (the range REwarp expects)."""
    img = Image.open(path).convert('RGB')
    tensor = transforms.ToTensor()(img).unsqueeze(0)  # [0, 1]
    tensor = tensor * 2.0 - 1.0                       # -> [-1, 1]
    return tensor


def linear_blender(ref, tgt, ref_m, tgt_m):
    """Copied verbatim from eval.py."""
    blur = GaussianBlur(kernel_size=(21, 21), sigma=20)
    r1, c1 = torch.nonzero(ref_m[0, 0], as_tuple=True)
    r2, c2 = torch.nonzero(tgt_m[0, 0], as_tuple=True)

    center1 = (r1.float().mean(), c1.float().mean())
    center2 = (r2.float().mean(), c2.float().mean())

    vec = (center2[0] - center1[0], center2[1] - center1[1])

    ovl = (ref_m * tgt_m).round()[:, 0].unsqueeze(1)
    ref_m_ = ref_m[:, 0].unsqueeze(1) - ovl
    r, c = torch.nonzero(ovl[0, 0], as_tuple=True)

    ovl_mask = torch.zeros_like(ref_m_).cuda()
    proj_val = (r - center1[0]) * vec[0] + (c - center1[1]) * vec[1]
    ovl_mask[ovl.bool()] = (proj_val - proj_val.min()) / (proj_val.max() - proj_val.min() + 1e-3)

    mask1 = (blur(ref_m_ + (1 - ovl_mask) * ref_m[:, 0].unsqueeze(1)) * ref_m + ref_m_).clamp(0, 1)
    mask2 = (1 - mask1) * tgt_m
    stit = ref * mask1 + tgt * mask2

    return stit


def stitch(img_ref_path, img_tgt_path, ckpt_path, out_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ---- Load REwarp (loads H_model and T_model from the same .pth) ----
    print(f'Loading checkpoint: {ckpt_path}')
    sv_file = torch.load(ckpt_path, map_location=device)

    T_model = models.make(sv_file['T_model'], load_sd=True).to(device)
    H_model = models.make(sv_file['H_model'], load_sd=True).to(device)

    T_model.eval()
    H_model.eval()

    num_params = utils.compute_num_params(H_model, text=False) + \
                 utils.compute_num_params(T_model, text=False)
    print(f'Model Params: {num_params}')

    # ---- Load images ----
    print(f'Loading {img_ref_path} and {img_tgt_path}...')
    ref_ = load_image(img_ref_path).to(device)   # [-1, 1], (1,3,H,W)
    tgt_ = load_image(img_tgt_path).to(device)

    # Match the two images to the same spatial size (use the larger of each dim)
    # REwarp assumes ref and tgt have equal H, W.
    h = max(ref_.shape[-2], tgt_.shape[-2])
    w = max(ref_.shape[-1], tgt_.shape[-1])
    if ref_.shape[-2] != h or ref_.shape[-1] != w:
        ref_ = F.interpolate(ref_, size=(h, w), mode='bilinear', align_corners=True)
    if tgt_.shape[-2] != h or tgt_.shape[-1] != w:
        tgt_ = F.interpolate(tgt_, size=(h, w), mode='bilinear', align_corners=True)
    b = ref_.shape[0]

    # Model runs at 512x512
    if h != 512 or w != 512:
        ref = F.interpolate(ref_, size=(512, 512), mode='bilinear', align_corners=True)
        tgt = F.interpolate(tgt_, size=(512, 512), mode='bilinear', align_corners=True)
    else:
        ref = ref_
        tgt = tgt_
    scale = h / ref.shape[-2]

    hcell_iter = 6
    tcell_iter = 3
    T_model.iters_lev = tcell_iter

    with torch.no_grad():
        # ---- H-Cell: homography estimation ----
        print('Running H-Cell (global homography)...')
        _, disps, hinp = H_model(ref, tgt, iters_lev0=hcell_iter)

        # Compute warped coordinates for the output canvas
        H, img_h, img_w, offset = utils.get_warped_coords(
            disps[-1], scale=(h / 512, w / 512), size=(h, w)
        )
        H_, *_ = utils.get_H(
            disps[-1].reshape(ref.shape[0], 2, -1).permute(0, 2, 1),
            [*ref.shape[-2:]]
        )
        H_ = utils.compens_H(H_, [*ref.shape[-2:]])

        grid = utils.make_coordinate_grid([*ref.shape[-2:]], type=H_.type())
        grid = grid.reshape(1, -1, 2).repeat(ref.shape[0], 1, 1)

        mesh_homography = utils.warp_coord(grid, H_.to(device)).reshape(b, *ref.shape[-2:], -1)
        ones = torch.ones_like(ref_).to(device)
        tgt_w = F.grid_sample(tgt, mesh_homography, align_corners=True)

        # ---- T-Cell: TPS refinement ----
        print('Running T-Cell (TPS refinement)...')
        flows = T_model(tgt_w, ref, iters=tcell_iter, scale=scale)

        # ---- Compose final stitched image ----
        print('Compositing stitched image...')
        translation = utils.get_translation(*offset)
        T_ref = translation.clone()
        T_tgt = torch.inverse(H).double() @ translation.to(device)

        sizes = (img_h, img_w)
        if img_h > 5000 or img_w > 5000:
            print(f'FAILURE: stitched canvas too large ({img_h}x{img_w}). '
                  f'This usually means the images have very low overlap or extreme parallax.')
            return

        coord1 = utils.to_pixel_samples(None, sizes=sizes).to(device)
        mesh_r, _ = utils.gridy2gridx_homography(
            coord1.contiguous(), *sizes, *tgt_.shape[-2:], T_ref.to(device), cpu=False
        )
        mesh_r = mesh_r.reshape(b, img_h, img_w, 2).to(device).flip(-1)

        coord2 = utils.to_pixel_samples(None, sizes=sizes).to(device)
        mesh_t, _ = utils.gridy2gridx_homography(
            coord2.contiguous(), *sizes, *tgt_.shape[-2:], T_tgt.to(device), cpu=False
        )
        mesh_t = mesh_t.reshape(b, img_h, img_w, 2).to(device).flip(-1)

        flow = flows[-1] / 511
        if flow.shape[-2] != 512 or flow.shape[-1] != 512:
            flow = F.interpolate(
                flow.permute(0, 3, 1, 2), size=(h, w), mode='bilinear'
            ).permute(0, 2, 3, 1) * 2

        mesh_t[:, offset[0]:offset[0] + h, offset[1]:offset[1] + w, :] += flow

        ref_w = F.grid_sample(ref_, mesh_r, mode='bilinear', align_corners=True)
        mask_r = F.grid_sample(ones, mesh_r, mode='nearest', align_corners=True)
        tgt_w = F.grid_sample(tgt_, mesh_t, mode='bilinear', align_corners=True)
        mask_t = F.grid_sample(ones, mesh_t, mode='nearest', align_corners=True)

        # Convert from [-1, 1] back to [0, 1]
        ref_w = (ref_w + 1) / 2 * mask_r
        tgt_w = (tgt_w + 1) / 2 * mask_t

        ovl = (mask_r * mask_t).round()
        if ovl.sum() == 0:
            print('FAILURE: no overlap between warped images.')
            return

        stit = linear_blender(ref_w, tgt_w, mask_r, mask_t)
        # Fill regions outside both images with white (paper uses black, but white reads better)
        stit += (1 - (mask_r + mask_t).clamp(0, 1))

    # ---- Save ----
    out_img = (stit[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(out_img).save(out_path)
    print(f'Stitched image saved to {out_path}')
    print(f'Output size: {img_h}x{img_w}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', required=True, help='Reference image path')
    parser.add_argument('--img2', required=True, help='Target image path')
    parser.add_argument('--out', default='stitched.jpg', help='Output path')
    parser.add_argument('--ckpt', default='save/_TCell/epoch-last.pth',
                        help='REwarp checkpoint path (contains both H_model and T_model)')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    stitch(args.img1, args.img2, args.ckpt, args.out)
