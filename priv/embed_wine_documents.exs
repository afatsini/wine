max_sequence_length = 120
batch_size = 128

{bert, bert_params} =
  AxonOnnx.import("priv/models/model.onnx", batch: batch_size, sequence: max_sequence_length)

bert = Axon.nx(bert, fn {_, out} -> out end)

{:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("bert-base-uncased")


path_to_wines = "priv/wine_documents.jsonl"
endpoint = "https://localhost:9200/wine/_doc/"
password = System.get_env("ELASTICSEARCH_PASSWORD")
credentials = "elastic:#{password}"
headers = [
  Authorization: "Basic #{Base.encode64(credentials)}",
  "Content-Type": "application/json"
]
options = [ssl: [cacertfile: "http_ca.crt"]]

document_stream =
  path_to_wines
  |> File.stream!()
  |> Stream.map(&Jason.decode!/1)
  |> Stream.map(fn document -> {document["url"], EmbedWineDocuments.format_document(document)} end)
  |> Stream.chunk_every(batch_size)
  |> Stream.flat_map(fn batches ->
    {urls, texts} = Enum.unzip(batches)
    inputs = EmbedWineDocuments.encode_text(tokenizer, texts, max_sequence_length)
    embedded = EmbedWineDocuments.compute_embedding(bert, bert_params, inputs)

    embedded
    |> Nx.to_batched(1)
    |> Enum.map(&Nx.to_flat_list(Nx.squeeze(&1)))
    |> Enum.zip_with(urls, fn vec, url -> %{"url" => url, "document-vector" => vec} end)
    |> Enum.map(&Jason.encode!/1)
  end)
  |> Stream.map(fn data ->
    {:ok, _} = HTTPoison.post(endpoint, data, headers, options)
    :ok
  end)

Enum.reduce(document_stream, 0, fn :ok, counter ->
  IO.write("\rDocuments Embedded: #{counter}")
  counter + 1
end)
