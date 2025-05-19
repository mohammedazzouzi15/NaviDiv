---
mode: 'agent'

tools: [ 'getReleaseFeatures', 'file_search', 'semantic_search', 'read_file', 'insert_edit_into_file', 'create_file', 'replace_string_in_file', 'fetch_webpage', 'vscode_search_extensions_internal']

---
# Project general coding standards

## Naming Conventions
- Use PascalCase for component names, interfaces, and type aliases
- Use camelCase for variables, functions, and methods
- Prefix private class members with underscore (_)
- Use ALL_CAPS for constants

## Error Handling
- Use try/catch blocks for async operations
- Implement proper error boundaries in React components
- Always log errors with contextual information

Go through the codebase and ensure that:
- All variable and function names follow the naming conventions
- All class members are properly prefixed with an underscore if they are private
- All constants are in ALL_CAPS
- All async operations are wrapped in try/catch blocks
- All React components have error boundaries
- All errors are logged with contextual information
- All comments are clear and concise
- All code is properly formatted and adheres to the project's style guide
- All unnecessary comments are removed
