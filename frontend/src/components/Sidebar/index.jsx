/**
 * Sidebar component re-export
 *
 * This module exports the main Sidebar component and its subcomponents.
 * Import with: import Sidebar from './components/Sidebar'
 */

export { default } from './Sidebar'
export { default as PromptSection } from './PromptSection'
export { default as FilterControls } from './FilterControls'
export { default as TagSearch } from './TagSearch'
export { default as useDebounce } from './hooks/useDebounce'

// Re-export constants from FilterControls
export { ALL_RATINGS, SORT_OPTIONS, MIN_AGE_LIMIT, MAX_AGE_LIMIT, TIMEFRAME_OPTIONS } from './FilterControls'
