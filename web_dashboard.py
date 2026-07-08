"""
Web气象预测看板 - 使用Flask提供Web界面
通过浏览器访问，无需X11图形环境
"""
import os
import json
import time
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

JSON_FILE = "prediction_output.json"
CONFIG_FILE = "config.json"


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/prediction')
def get_prediction():
    try:
        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({'success': True, 'data': data})
        else:
            return jsonify({'success': False, 'error': '预测文件不存在'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/config')
def get_config():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({'success': True, 'data': data})
        else:
            return jsonify({'success': False, 'error': '配置文件不存在'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/status')
def get_status():
    return jsonify({
        'success': True,
        'data': {
            'server_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'json_file': JSON_FILE,
            'json_exists': os.path.exists(JSON_FILE),
            'json_mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(JSON_FILE))) if os.path.exists(JSON_FILE) else None
        }
    })


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Web气象预测看板')
    parser.add_argument('-p', '--port', type=int, default=5000, help='端口号')
    parser.add_argument('-H', '--host', default='0.0.0.0', help='监听地址')
    parser.add_argument('-f', '--file', default='prediction_output.json', help='JSON文件路径')
    args = parser.parse_args()
    
    JSON_FILE = args.file
    
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if os.path.exists(template_dir):
        app.template_folder = template_dir
    
    print(f"Web看板启动: http://{args.host}:{args.port}")
    print(f"监听文件: {JSON_FILE}")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)