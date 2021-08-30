/// @file
#version 460 core

uniform mat4 V_mat;                                                             // View matrix.
uniform mat4 P_mat;                                                             // Projection matrix.
uniform float size_x;                                                           // Framebuffer size_x.
uniform float size_y;                                                           // Framebuffer size_y.
uniform float AR;                                                               // Framebuffer aspect ratio.

in vec4 color;                                                                  // Fragment color.
in vec2 quad;                                                                   // Billboard quad UV coordinates.
in float AR_quad;                                                               // Billboard quad aspect ratio.

out vec4 fragment_color;                                                        // Fragment color.

void main(void)
{
  float u_P = -0.5*AR_quad + 0.5;                                               // Central node billboard U coordinate.
  float u_Q = 0.5*AR_quad - 0.5;                                                // Neighbour node billboard U coordinate.
  vec2  P = vec2(u_P, 0.0) - quad;                                              // Central node billboard UV vector.
  vec2  Q = vec2(u_Q, 0.0) - quad;                                              // Neighbour node billboard UV vector.
  float R_P = length(P);                                                        // Central node radial coordinate.
  float R_Q = length(Q);                                                        // Neighbour node radial coordinate.
  float coulomb_P = 1.0/R_P;                                                    // P Coulomb potential.
  float coulomb_Q = 1.0/R_Q;                                                    // Q Coulomb potential.
  float string_PQ = 2.0*log((Q.x + R_Q)/(P.x + R_P));                           // PQ string potential.
  float f;                                                                      // PQ metaball potential.

  f = coulomb_P + coulomb_Q + string_PQ;

  float bloom = 0.1;                                                            // Blooming radius.

  float k1;                                                                     // Blooming coefficient.
  float k2;                                                                     // Smoothness coefficient.
  float k3;                                                                     // Smoothness coefficient.
  
  k1 = 1.0 - smoothstep(0.0, bloom, 1/f);                                       // Computing smoothness coefficient...
  k2 = 1.0 - smoothstep(0.0, 0.1, 1/f);                                         // Computing smoothness coefficient...
  k3 = 1.0 - smoothstep(0.2, 0.3, 1/f);                                         // Computing smoothness coefficient...

  if (k1 == 0.0)
  {
    discard;                                                                    // Discarding fragment point...
  }

  fragment_color = vec4(color.rgb, k1*color.a);
  //fragment_color = vec4(0.4*vec3(k2, 1.1*k3, k1) + color.rgb, 0.0 + k1);        // Setting fragment color...
}