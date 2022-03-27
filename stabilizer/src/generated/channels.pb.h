/* Automatically generated nanopb header */
/* Generated by nanopb-0.4.5 */

#ifndef PB_CHANNELS_PB_H_INCLUDED
#define PB_CHANNELS_PB_H_INCLUDED
#include <pb.h>

#if PB_PROTO_HEADER_VERSION != 40
#error Regenerate this file with the current version of nanopb generator.
#endif

/* Struct definitions */
/* These values will always be between -255 and 255 to correspond to Arduino's
 analogWrite(). The negative values are for rotating the motor backwards. */
typedef struct _MotorSpeed { 
    /* This field is declared optional so the overall message always takes space
 on the wire. If it was not optional and both fields were set to 0, the
 serialized packet would take up 0 bytes on the wire. The Arduino UDP
 library does not notify us when we receive a zero-length UDP packet, so we
 must keep the serialized message >= 1 byte in length so we can detect it
 even when both fields are set to 0. */
    bool has_up;
    int32_t up; 
    int32_t down; 
    int32_t left; 
    int32_t right; 
    int32_t front; 
    int32_t back; 
} MotorSpeed;


#ifdef __cplusplus
extern "C" {
#endif

/* Initializer values for message structs */
#define MotorSpeed_init_default                  {false, 0, 0, 0, 0, 0, 0}
#define MotorSpeed_init_zero                     {false, 0, 0, 0, 0, 0, 0}

/* Field tags (for use in manual encoding/decoding) */
#define MotorSpeed_up_tag                        1
#define MotorSpeed_down_tag                      2
#define MotorSpeed_left_tag                      3
#define MotorSpeed_right_tag                     4
#define MotorSpeed_front_tag                     5
#define MotorSpeed_back_tag                      6

/* Struct field encoding specification for nanopb */
#define MotorSpeed_FIELDLIST(X, a) \
X(a, STATIC,   OPTIONAL, SINT32,   up,                1) \
X(a, STATIC,   SINGULAR, SINT32,   down,              2) \
X(a, STATIC,   SINGULAR, SINT32,   left,              3) \
X(a, STATIC,   SINGULAR, SINT32,   right,             4) \
X(a, STATIC,   SINGULAR, SINT32,   front,             5) \
X(a, STATIC,   SINGULAR, SINT32,   back,              6)
#define MotorSpeed_CALLBACK NULL
#define MotorSpeed_DEFAULT NULL

extern const pb_msgdesc_t MotorSpeed_msg;

/* Defines for backwards compatibility with code written before nanopb-0.4.0 */
#define MotorSpeed_fields &MotorSpeed_msg

/* Maximum encoded size of messages (where known) */
#define MotorSpeed_size                          36

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
