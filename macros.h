
#define _STR_CONCAT(a,b) a##b
#define  STR_CONCAT(a,b) _STR_CONCAT(a,b)

/* -------------------------------------------------------------------------- */

#define  SCALAR_REFLECT_TYPE STR_CONCAT(SCALAR_NAME, Type)

#define  NEW_SCALAR STR_CONCAT(New,  SCALAR_NAME)
#define NULL_SCALAR STR_CONCAT(Null, SCALAR_NAME)

#define  NEW_VECTOR STR_CONCAT(New,  VECTOR_NAME)
#define NULL_VECTOR STR_CONCAT(Null, VECTOR_NAME)
#define  NIL_VECTOR STR_CONCAT(nil,  VECTOR_NAME)
#define   AS_VECTOR STR_CONCAT(As,   VECTOR_NAME)

#define  NEW_MATRIX STR_CONCAT(New,  MATRIX_NAME)
#define NULL_MATRIX STR_CONCAT(Null, MATRIX_NAME)
#define  NIL_MATRIX STR_CONCAT(nil,  MATRIX_NAME)
#define   AS_MATRIX STR_CONCAT(As,   MATRIX_NAME)
