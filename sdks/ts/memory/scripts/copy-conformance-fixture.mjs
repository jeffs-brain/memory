// SPDX-License-Identifier: Apache-2.0

import { copyFile, mkdir } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const packageDir = resolve(here, '..')
const source = resolve(packageDir, '../../../spec/conformance/http-contract.json')
const target = resolve(packageDir, 'dist/conformance/http-contract.json')

await mkdir(dirname(target), { recursive: true })
await copyFile(source, target)
