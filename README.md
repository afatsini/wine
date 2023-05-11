# Wine

Based on https://dockyard.com/blog/2022/09/28/semantic-search-with-phoenix-axon-and-elastic

python dependencies:

```
asdf plugin-add python
asdf install python 3.9.16

asdf local python 3.9.16

pip install transformers
pip install tensorflow
pip install onnx
pip install tf2onnx
pip install onnxruntime
python -m transformers.onnx --model=bert-base-uncased priv/models/

```