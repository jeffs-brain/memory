# RN Hello World

Minimal Expo example for `@jeffs-brain/memory-react-native` using the supported cloud-backed path.

## Run

From the repo root:

```bash
bun run build:packages
cd examples/rn/hello-world
bun install
EXPO_PUBLIC_OPENAI_API_KEY=... bun run start
```

Optional environment variables:

- `EXPO_PUBLIC_OPENAI_BASE_URL`
- `EXPO_PUBLIC_OPENAI_MODEL`
- `EXPO_PUBLIC_OPENAI_EMBED_MODEL`
- `EXPO_PUBLIC_MEMORY_BRAIN_ID`

## Notes

- This example uses Expo file storage plus `@op-engineering/op-sqlite` for the on-device brain and search index.
- Chat, extract, and reflect require a configured OpenAI-compatible provider.
- Local on-device inference is not shown here. That path still requires the app to register native inference and embedding modules.
- Seed the sample memory first if you want an immediate recall result before chatting.

