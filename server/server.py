# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 19:01:45 2024

@author: Jenosis225
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载保存的模型
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 推断函数
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.item()

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        data = json.loads(post_data)

        sample_texts = data.get("sample_texts", "")
        print(f"Received from client: {sample_texts}")

        # 做出預測
        prediction = predict(sample_texts)
        response_message = "高風險" if prediction == 1 else "低風險"

        # 發送回應
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = {'message': response_message}
        self.wfile.write(json.dumps(response).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ('', 5000)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {server_address[0]}:{server_address[1]}...')
    httpd.serve_forever()

if __name__ == "__main__":
    run()
