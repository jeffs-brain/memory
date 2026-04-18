/**
 * OpenFGA adapter for `@jeffs-brain/memory`.
 *
 * Optional sibling package that implements the `AccessControlProvider`
 * contract from `@jeffs-brain/memory/acl` against an OpenFGA HTTP API.
 */

export {
  createOpenFgaProvider,
  OpenFgaHttpError,
  OpenFgaRequestError,
  type OpenFgaOptions,
  type FetchLike,
} from './openfga.js'
