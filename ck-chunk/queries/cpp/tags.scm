; C++ chunk definitions

; Functions
(function_definition) @definition.function

; Classes, structs, enums, and unions
(class_specifier) @definition.class
(struct_specifier) @definition.struct
(enum_specifier) @definition.enum
(union_specifier) @definition.class

; Namespaces
(namespace_definition) @module.namespace

; Templates
(template_declaration) @definition.text

; Typedefs and type aliases
(type_definition) @definition.text
(alias_declaration) @definition.text

; Top-level declarations
(declaration) @definition.text

; Preprocessor macros
(preproc_function_def) @definition.function
(preproc_def) @definition.text
