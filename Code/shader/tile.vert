/// @file   tile.vert
/// @author Erik ZORZIN
/// @date   23JUL2020
/// @brief  This vertex file creates a tile out of a 4D point coordinates.

#version 410 core

const vec3 A = vec3(-1.0, -1.0, +0.0);                                                              ///< @brief **Vertex "A".**
const vec3 B = vec3(+1.0, -1.0, +0.0);                                                              ///< @brief **Vertex "B".**
const vec3 C = vec3(-1.0, +1.0, +0.0);                                                              ///< @brief **Vertex "C".**
const vec3 D = vec3(+1.0, +1.0, +0.0);                                                              ///< @brief **Vertex "D".**
const vec3 n = vec3(+0.0, +0.0, +1.0);                                                              ///< @brief **Face normal.**
const float s = 0.008;                                                                              ///< @brief **Tile side.**
const vec3 l = vec3(0.0, -1.0, 0.0);                                                                ///< @brief **Light direction.**

layout (location = 0) in vec4   tile_center;                                                        ///< @brief **Tile center.**
layout (location = 1) in vec4   tile_color;                                                         ///< @brief **Tile color.**

out VS_OUT
{
  vec4 vertex_A;                                                                                    ///< Vertex "A".
  vec4 vertex_B;                                                                                    ///< Vertex "B".
  vec4 vertex_C;                                                                                    ///< Vertex "C".
  vec4 vertex_D;                                                                                    ///< Vertex "D".
  vec4 color_front;                                                                                 ///< Face "ABDC" front color.
  vec4 color_back;                                                                                  ///< Face "ABDC" back color.
  
} vs_out;

uniform mat4 V_mat;                                                                                 ///< View matrix.
uniform mat4 P_mat;                                                                                 ///< Projection matrix.

/// @function main
/// Computing rendering point coordinates:
void main(void)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////// TILE: 2D binary square //////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  //       (-1.0, +1.0, +0.0)  C--------D  (+1.0, +1.0, +0.0)
  //                           |        |
  //                           |        |
  //       (-1.0, -1.0, +0.0)  A--------B  (+1.0, -1.0, +0.0)
  //
  //                               y (points up)
  //                               |
  //                               o -- x (points right)
  //                              /
  //                             z (points out of the screen)
  //
  //                               UP
  //                               |  / BACK
  //                               | /
  //                     LEFT -----+----- RIGHT
  //                              /|
  //                       FRONT / |
  //                              DOWN
  //
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  vec3 light;                                                                                       // Light direction.
  vec3 normal;                                                                                      // Face "ABDC" normal.
  float diffusion_front;                                                                            // Face "ABDC" front diffusion coefficient.
  float diffusion_back;                                                                             // Face "ABDC" back diffusion coefficient.

  gl_Position = P_mat*V_mat*tile_center;                                                            // Setting tile position...
  light = -normalize(l);                                                                            // Normalizing and inverting light direction...

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// TILE'S FACE BARICENTRIC NORMALS ///////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  normal = vec3(P_mat*V_mat*(tile_center + vec4(n, +1.0)));                                         // Computing face "ABDC" normal.
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////// TILE'S FACE DIFFUSION COEFFICIENTS ////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  diffusion_front = clamp(dot(light, normal), 0.3, 1.0);                                            // Computing face "ABDC" front diffusion coefficient.
  diffusion_back = clamp(dot(light, -normal), 0.3, 1.0);                                            // Computing face "ABDC" back diffusion coefficient.
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////// TILE'S NU_VERTEX BARICENTRIC COORDINATES /////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  vs_out.vertex_A = P_mat*V_mat*(tile_center + vec4(s*A, 1.0));                                     // Computing vertex "A".
  vs_out.vertex_B = P_mat*V_mat*(tile_center + vec4(s*B, 1.0));                                     // Computing vertex "B".
  vs_out.vertex_C = P_mat*V_mat*(tile_center + vec4(s*C, 1.0));                                     // Computing vertex "C".
  vs_out.vertex_D = P_mat*V_mat*(tile_center + vec4(s*D, 1.0));                                     // Computing vertex "D".

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////// TILE'S FACE COLORS ///////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  vs_out.color_front = vec4(diffusion_front*vec3(tile_color), 1.0);                                 // Computing face "ABDC" front color.
  vs_out.color_back = vec4(diffusion_back*vec3(tile_color), 1.0);                                   // Computing face "ABDC" back color.
}
