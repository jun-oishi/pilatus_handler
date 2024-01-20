
## C++製モジュールのコンパイルのための環境構築

依存するパッケージ

- g++ (c++23を使用)
- make
- pybind11 (PythonとC++のインターフェース)
- python3.11-dev (Python.hを使用)
- eigen

インストール

```bash
sudo apt install -y g++ make python3.11-dev libeigen3-dev
pip install pybind11 # pyproject.tomlに記述してあるので、pip install . すれば不要
```