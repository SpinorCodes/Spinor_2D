/// @file   tile.geom
/// @author Erik ZORZIN
/// @date   23JUL2020
/// @brief  This geometry file creates a tile out of a 4D point coordinates.

#version 410 core

layout (points) in;                                                                                 ///< @brief **Input points.**
layout (triangle_strip, max_vertices = 26) out;                                                     ///< @brief **Output points.**

in VS_OUT
{
  vec4 vertex_A;                                                                                    ///< Vertex "A".
  vec4 vertex_B;                                                                                    ///< Vertex "B".
  vec4 vertex_C;                                                                                    ///< Vertex "C".
  vec4 vertex_D;                                                                                    ///< Vertex "D".
  vec4 color_front;                                                                                 ///< Face "ABDC" front color.
  vec4 color_back;                                                                                  ///< Face "ABDC" back color.
} gs_in[];

out vec4 tile_color;                                                                                ///< @brief **Tile color (for fragment shader).**

/// @function main
/// Generating voxel verteces:
void main()
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// VOXEL: triangulation scheme //////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  //                           C--------D
  //                           |        |
  //                           |        |
  //                           A--------B
  //
  //  Triangle strip sequence:
  //
  //             ABC +
  //                 | 
  //           (BC)D +
  //
  //             ACB +
  //                 | 
  //           (CB)D +
  //          
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////// ABC ///////////////////////////////////////////////
  tile_color = gs_in[0].color_front;                                                                // Setting tile color...
  gl_Position = gs_in[0].vertex_A;                                                                  // Setting tile position...
  EmitVertex();                                                                                     // "A" vertex.

  tile_color = gs_in[0].color_front;                                                                // Setting tile color...
  gl_Position = gs_in[0].vertex_B;                                                                  // Setting tile position...
  EmitVertex();                                                                                     // "B" vertex.

  tile_color = gs_in[0].color_front;                                                                // Setting tile color...
  gl_Position = gs_in[0].vertex_C;                                                                  // Setting tile position...
  EmitVertex();                                                                                     // "C" vertex.

  tile_color = gs_in[0].color_front;                                                                // Setting tile color...
  gl_Position = gs_in[0].vertex_D;                                                                  // Setting tile position...
  EmitVertex();                                                                                     // "D" vertex.

  EndPrimitive();

  tile_color = gs_in[0].color_back;                                                                 // Setting tile color...
  gl_Position = gs_in[0].vertex_A;                                                                  // Setting tile position...
  EmitVertex();                                                                                     // "A" vertex.

  tile_color = gs_in[0].color_back;                                                                 // Setting tile color...
  gl_Position = gs_in[0].vertex_C;                                                                  // Setting tile position...
  EmitVertex();                                                                                     // "C" vertex.

  tile_color = gs_in[0].color_back;                                                                 // Setting tile color...
  gl_Position = gs_in[0].vertex_B;                                                                  // Setting tile position...
  EmitVertex();                                                                                     // "B" vertex.

  tile_color = gs_in[0].color_back;                                                                 // Setting tile color...
  gl_Position = gs_in[0].vertex_D;                                                                  // Setting tile position...
  EmitVertex();                                                                                     // "D" vertex.

  EndPrimitive();
}
