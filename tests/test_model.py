import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.train import Infer


def test_predict():
    # Test the predict function
    samples = [
        "id=1 OR 1=1",  # SQLi
        "<script>alert('x')</script>",  # XSS
        "page=home",  # 正常请求
        "script",  # 正常请求
    ]
    max_len = 100
    model_path = f"model_save"  # Path to your model file
    infer = Infer(f"{model_path}/model.pt")
    for text in samples:
        print(infer.predict(text, max_len=max_len))


if __name__ == "__main__":
    test_predict()
