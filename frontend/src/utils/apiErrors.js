export function shouldSuppressOptionalNotFound(config, status) {
  return config?.suppressErrorToast === true && status === 404
}
