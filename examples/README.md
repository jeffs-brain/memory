# Examples

| Language   | Path                              | Purpose                                                     |
| ---------- | --------------------------------- | ----------------------------------------------------------- |
| TypeScript | [ts/hello-world](./ts/hello-world) | Markdown chunk + BM25 search via `createSearchIndex`        |
| React Native | [rn/hello-world](./rn/hello-world) | Expo app using the RN SDK with local brain storage plus a cloud-backed provider |
| Go         | [go/hello-world](./go/hello-world) | BM25 search via `knowledge.Ingest` + `knowledge.Search`     |
| Python     | [py/hello-world](./py/hello-world) | BM25 search via `knowledge.new` + `kb.ingest` + `kb.search` |

Each example is a self-contained mini-project; see its README for run instructions.

## Running the whole matrix

The cross-SDK eval runner at [`../eval`](../eval) drives every SDK through the shared HTTP ask contract and records smoke / LME results under `eval/results/`. Use that when you want an apples-to-apples benchmark across TypeScript, Go, and Python.
