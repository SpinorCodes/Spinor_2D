/// @file   tile.frag
/// @author Erik ZORZIN
/// @date   23JUL2020
/// @brief  This fragment file creates a tile out of a 4D point coordinates.

#version 410 core

in  vec4 tile_color;                                                                                // Tile color.
out vec4 fragment_color;                                                                            // Fragment color.

/// @function main
/// Setting tile color:
void main(void)
{
  fragment_color = tile_color;                                                                     // Setting fragment color...
}
