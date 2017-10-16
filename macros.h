
#define _STR_CONCAT(a,b) a##b
#define  STR_CONCAT(a,b) _STR_CONCAT(a,b)

/* -------------------------------------------------------------------------- */

#define  NEW_SCALAR STR_CONCAT(New,  SCALAR_TYPE)
#define NULL_SCALAR STR_CONCAT(Null, SCALAR_TYPE)

#define  NEW_VECTOR STR_CONCAT(New,  VECTOR_TYPE)
#define NULL_VECTOR STR_CONCAT(Null, VECTOR_TYPE)
#define  NIL_VECTOR STR_CONCAT(nil,  VECTOR_TYPE)

#define  NEW_MATRIX STR_CONCAT(New,  MATRIX_TYPE)
#define NULL_MATRIX STR_CONCAT(Null, MATRIX_TYPE)
#define  NIL_MATRIX STR_CONCAT(nil,  MATRIX_TYPE)

