import json
import base64
import os

def extract_images():
    with open('pset1.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 这些图片是按照你在 notebook 中从上到下生成图片的顺序排列的
    filenames = [
        "ae_results.png",    # AE重构
        "kl_vs_sgvb.png",    # KL vs SGVB 曲线
        "vae_sgvb.png",      # VAE SGVB 重构
        "vae_kl.png",        # VAE KL 重构
        "torus_orig.png",    # Torus 原始数据(可能不需要放到报告里，但先抽出来)
        "torus_recon.png",   # Torus 重构
        "torus_interp.png"   # Torus 线性插值
    ]

    image_idx = 0
    for cell in nb.get('cells', []):
        if 'outputs' in cell:
            for out in cell['outputs']:
                if 'data' in out and 'image/png' in out['data']:
                    png_data = out['data']['image/png']
                    # 有些时候 notebook 里的 base64 是一组字符串，有些是一个长字符串
                    if isinstance(png_data, list):
                        png_data = "".join(png_data)
                    
                    if image_idx < len(filenames):
                        filename = filenames[image_idx]
                    else:
                        filename = f"extra_img_{image_idx}.png"
                        
                    with open(filename, 'wb') as img_f:
                        img_f.write(base64.b64decode(png_data))
                    print(f"Saved {filename}")
                    
                    image_idx += 1

if __name__ == "__main__":
    extract_images()
