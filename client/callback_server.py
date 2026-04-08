from flask import Flask, request

app = Flask(__name__)

@app.route('/callback', methods=['POST'])
def callback():
    data = request.json
    print("\n===== 收到回调 =====")
    print(data)
    print("===================\n")
    return {"status": "received"}, 200

if __name__ == '__main__':
    app.run(port=9000)