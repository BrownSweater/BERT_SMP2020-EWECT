import torch
import argparse
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--input', type=str, help='input text')
    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
    parser.add_argument('--model_name', default='bert-base-chinese', type=str,
                        help='huggingface transformer model name')
    parser.add_argument('--model_path', default='workspace/wb/best.pt', type=str, help='model path')
    parser.add_argument('--num_labels', default=6, type=int, help='fine-tune num labels')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    print(f'Loading checkpoint: {args.model_path} ...')
    checkpoint = torch.load(args.model_path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=True)
    print(f'missing_keys: {missing_keys}\n'
                f'===================================================================\n')
    print(f'unexpected_keys: {unexpected_keys}\n'
                f'===================================================================\n')

    label = {0: 'happy', 1: 'angry', 2: 'sad', 3: 'fear', 4: 'surprise', 5: 'neutral'}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    token = tokenizer(args.input, padding='max_length', truncation=True, max_length=140)
    input_ids = torch.tensor(token['input_ids']).unsqueeze(0)
    model.eval()
    model.to(device)
    input_ids.to(device)
    with torch.no_grad():
        output = model(input_ids)
    pred = np.argmax(output.logits.detach().cpu().numpy(), axis=1).tolist()[0]
    print(f'##################### result: {label[pred]} #####################')


if __name__ == '__main__':
    main()