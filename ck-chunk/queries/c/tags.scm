; C chunk definitions

; Functions
(function_definition) @definition.function

; Structs, enums, and unions
(struct_specifier) @definition.struct
(enum_specifier) @definition.enum
(union_specifier) @definition.class

; Typedefs and declarations
(type_definition) @definition.text
(declaration) @definition.text

; Preprocessor macros
(preproc_function_def) @definition.function
(preproc_def) @definition.text
