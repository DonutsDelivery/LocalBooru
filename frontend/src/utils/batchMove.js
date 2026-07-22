export async function loadMoveDirectoryOptions(fetchDirectories) {
  const response = await fetchDirectories()
  if (!Array.isArray(response?.directories)) {
    throw new Error('Directory service returned an invalid response')
  }
  return response.directories
}
