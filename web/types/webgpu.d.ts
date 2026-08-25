// WebGPU is not in TypeScript's DOM lib yet, so pull the official types in
// globally rather than adding "@webgpu/types" to compilerOptions.types, which
// would override the automatic @types resolution Next.js relies on.
/// <reference types="@webgpu/types" />
