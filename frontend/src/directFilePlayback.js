export function createDirectFileItem(request) {
  if (!request?.direct_file || !request.id || !request.file_path || !request.filename
    || !request.url || !request.direct_file_token) {
    throw new Error('Invalid direct media file request')
  }
  return {
    ...request,
    original_filename: request.original_filename || request.filename,
    directory_id: null,
    library_id: null,
    is_local_direct_file: true,
    tags: [],
  }
}
