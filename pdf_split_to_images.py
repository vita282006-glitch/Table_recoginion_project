from pathlib import Path
import random
import fitz
import argparse


def split_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 200):
    doc = fitz.open(pdf_path)
    images = []
    for page_number, page in enumerate(doc, start=1):
        rect = page.rect
        half_width = rect.width / 2

        left_rect = fitz.Rect(rect.x0, rect.y0, rect.x0 + half_width, rect.y1)
        right_rect = fitz.Rect(rect.x0 + half_width, rect.y0, rect.x1, rect.y1)

        for side, clip_rect in [('left', left_rect), ('right', right_rect)]:
            img_name = f"{pdf_path.stem.replace(' ', '_')}_page_{page_number}_{side}.png"
            images.append((page, clip_rect, output_dir / img_name))

    for page, clip_rect, output_path in images:
        pix = page.get_pixmap(clip=clip_rect, matrix=fitz.Matrix(dpi / 72, dpi / 72))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(output_path)
    return [output_path for _, _, output_path in images]


def split_train_test(image_paths, train_ratio=0.8, seed=42):
    random.Random(seed).shuffle(image_paths)
    split_index = int(len(image_paths) * train_ratio)
    return image_paths[:split_index], image_paths[split_index:]


def main():
    parser = argparse.ArgumentParser(description='Convert PDFs to left/right split images and save train/test sets.')
    parser.add_argument('--pdf-dir', type=Path, default=Path('data/row/Дуга'), help='Directory with PDF files.')
    parser.add_argument('--train-dir', type=Path, default=Path('data/row/Дуга/train'), help='Output directory for train images.')
    parser.add_argument('--test-dir', type=Path, default=Path('data/row/Дуга/test'), help='Output directory for test images.')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio (0.0-1.0).')
    parser.add_argument('--dpi', type=int, default=200, help='Image DPI for PDF rendering.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for train/test split.')
    args = parser.parse_args()

    pdf_dir = args.pdf_dir
    train_dir = args.train_dir
    test_dir = args.test_dir
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    all_images = []
    pdf_paths = sorted({
        *pdf_dir.glob('*.PDF'),
        *pdf_dir.glob('*.pdf'),
    })
    for pdf_path in pdf_paths:
        print(f'Processing {pdf_path.name}')
        images = split_pdf_to_images(pdf_path, pdf_dir, dpi=args.dpi)
        all_images.extend(images)

    train_images, test_images = split_train_test(all_images, train_ratio=args.train_ratio, seed=args.seed)

    for src in train_images:
        dst = train_dir / src.name
        src.replace(dst)
    for src in test_images:
        dst = test_dir / src.name
        src.replace(dst)

    print(f'Total images: {len(all_images)}')
    print(f'Train images: {len(train_images)} saved to {train_dir}')
    print(f'Test images: {len(test_images)} saved to {test_dir}')


if __name__ == '__main__':
    main()
