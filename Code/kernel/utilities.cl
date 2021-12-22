/// @file     utilities.cl
/// @author   Erik ZORZIN
/// @date     26MAR2021
/// @brief    Some useful functions.
/// @details  Norm, Colormap.

// Normalizes a float3 vector. Returns a zero vector if its norm is too small.
float3 normzero3 (float3 v)
{
  float3 v_norm;                                                                  // Norm.

  if(length(v) > FLT_EPSILON)
  {
    v_norm = normalize(v);                                                        // Computing norm...
  }
  else
  {
    v_norm = (float3)(0.0f, 0.0f, 0.0f);                                          // Resetting norm...
  }

  return v_norm;                                                                  // Returning norm...
}

// Adjusts float to zero if too small.
float adjzero (float v)
{
  float v_adj;                                                                    // Adjusted value.

  if(fabs(v) > FLT_EPSILON)
  {
    v_adj = v;                                                                    // Setting value...
  }
  else
  {
    v_adj = 0.0f;                                                                 // Retting value...
  }

  return v_adj;                                                                   // Returning adjusted value...
}

// Adjusts float3 components to zero if they are too small.
float3 adjzero3 (float3 v)
{
  float3 v_adj;                                                                     // Adjusted float3.

  v_adj.x = adjzero(v.x);                                                           // Adjusting component value...
  v_adj.y = adjzero(v.y);                                                           // Adjusting component value...
  v_adj.z = adjzero(v.z);                                                           // Adjusting component value...

  return v_adj;                                                                     // Returning adjusted float3.
}

// Multiplies two float numbers. Returns zero if at least one of the two is too small, or is their product is too small.
float mulzero (float a, float b)
{
  float c;                                                                          // Adjusted float product.

  c = a*b;                                                                          // Computing product...

  if(
      (fabs(a) < FLT_EPSILON) ||
      (fabs(b) < FLT_EPSILON) ||
      (fabs(c) < FLT_EPSILON)
    )
  {
    c = 0.0f;                                                                       // Resetting product...
  }

  return c;                                                                         // Returning product...
}

// Multiplies a float scalar A by a float3 vector V. Sets the individual components of the product to zero if 
// A is too small, or if the individual components of V are too small, or if the individual product components
// are too small.
float3 mulzero3 (float a, float3 v)
{
  float3 c;                                                                         // Adjusted float3 product.

  c.x = mulzero(a, v.x);                                                            // Computing component product...
  c.y = mulzero(a, v.y);                                                            // Computing component product...
  c.z = mulzero(a, v.z);                                                            // Computing component product...

  return c;                                                                         // Returning product...
}

float pownzero (float a, int n)
{
  float p;                                                                          // Power.

  p = pown(a, n);                                                                   // Computing power...

  if(
      (fabs(a) < FLT_EPSILON) ||
      (fabs(p) < FLT_EPSILON)
    )
  {
    p = 0.0f;                                                                       // Resetting power...
  }

  return p;                                                                         // Returning power...
}

// Computes the reciprocal of a float number.
float recipzero (float a)
{
  float b;

  if(fabs(a) < FLT_EPSILON)
  {
    b = FLT_MAX;
  }
  else
  {
    b = 1.0f/a;

    if(fabs(b) < FLT_EPSILON)
    {
      b = 0.0f;
    }
  }

  return b;
}

float3 colormap (float intensity)
{
    float3       turbo_colormap[256];
    unsigned int i;

    turbo_colormap[0] = (float3)(0.18995f, 0.07176f, 0.23217f);
    turbo_colormap[1] = (float3)(0.19483f, 0.08339f, 0.26149f);
    turbo_colormap[2] = (float3)(0.19956f, 0.09498f, 0.29024f);
    turbo_colormap[3] = (float3)(0.20415f, 0.10652f, 0.31844f);
    turbo_colormap[4] = (float3)(0.20860f, 0.11802f, 0.34607f);
    turbo_colormap[5] = (float3)(0.21291f, 0.12947f, 0.37314f);
    turbo_colormap[6] = (float3)(0.21708f, 0.14087f, 0.39964f);
    turbo_colormap[7] = (float3)(0.22111f, 0.15223f, 0.42558f);
    turbo_colormap[8] = (float3)(0.22500f, 0.16354f, 0.45096f);
    turbo_colormap[9] = (float3)(0.22875f, 0.17481f, 0.47578f);
    turbo_colormap[10] = (float3)(0.23236f, 0.18603f, 0.50004f);
    turbo_colormap[11] = (float3)(0.23582f, 0.19720f, 0.52373f);
    turbo_colormap[12] = (float3)(0.23915f, 0.20833f, 0.54686f);
    turbo_colormap[13] = (float3)(0.24234f, 0.21941f, 0.56942f);
    turbo_colormap[14] = (float3)(0.24539f, 0.23044f, 0.59142f);
    turbo_colormap[15] = (float3)(0.24830f, 0.24143f, 0.61286f);
    turbo_colormap[16] = (float3)(0.25107f, 0.25237f, 0.63374f);
    turbo_colormap[17] = (float3)(0.25369f, 0.26327f, 0.65406f);
    turbo_colormap[18] = (float3)(0.25618f, 0.27412f, 0.67381f);
    turbo_colormap[19] = (float3)(0.25853f, 0.28492f, 0.69300f);
    turbo_colormap[20] = (float3)(0.26074f, 0.29568f, 0.71162f);
    turbo_colormap[21] = (float3)(0.26280f, 0.30639f, 0.72968f);
    turbo_colormap[22] = (float3)(0.26473f, 0.31706f, 0.74718f);
    turbo_colormap[23] = (float3)(0.26652f, 0.32768f, 0.76412f);
    turbo_colormap[24] = (float3)(0.26816f, 0.33825f, 0.78050f);
    turbo_colormap[25] = (float3)(0.26967f, 0.34878f, 0.79631f);
    turbo_colormap[26] = (float3)(0.27103f, 0.35926f, 0.81156f);
    turbo_colormap[27] = (float3)(0.27226f, 0.36970f, 0.82624f);
    turbo_colormap[28] = (float3)(0.27334f, 0.38008f, 0.84037f);
    turbo_colormap[29] = (float3)(0.27429f, 0.39043f, 0.85393f);
    turbo_colormap[30] = (float3)(0.27509f, 0.40072f, 0.86692f);
    turbo_colormap[31] = (float3)(0.27576f, 0.41097f, 0.87936f);
    turbo_colormap[32] = (float3)(0.27628f, 0.42118f, 0.89123f);
    turbo_colormap[33] = (float3)(0.27667f, 0.43134f, 0.90254f);
    turbo_colormap[34] = (float3)(0.27691f, 0.44145f, 0.91328f);
    turbo_colormap[35] = (float3)(0.27701f, 0.45152f, 0.92347f);
    turbo_colormap[36] = (float3)(0.27698f, 0.46153f, 0.93309f);
    turbo_colormap[37] = (float3)(0.27680f, 0.47151f, 0.94214f);
    turbo_colormap[38] = (float3)(0.27648f, 0.48144f, 0.95064f);
    turbo_colormap[39] = (float3)(0.27603f, 0.49132f, 0.95857f);
    turbo_colormap[40] = (float3)(0.27543f, 0.50115f, 0.96594f);
    turbo_colormap[41] = (float3)(0.27469f, 0.51094f, 0.97275f);
    turbo_colormap[42] = (float3)(0.27381f, 0.52069f, 0.97899f);
    turbo_colormap[43] = (float3)(0.27273f, 0.53040f, 0.98461f);
    turbo_colormap[44] = (float3)(0.27106f, 0.54015f, 0.98930f);
    turbo_colormap[45] = (float3)(0.26878f, 0.54995f, 0.99303f);
    turbo_colormap[46] = (float3)(0.26592f, 0.55979f, 0.99583f);
    turbo_colormap[47] = (float3)(0.26252f, 0.56967f, 0.99773f);
    turbo_colormap[48] = (float3)(0.25862f, 0.57958f, 0.99876f);
    turbo_colormap[49] = (float3)(0.25425f, 0.58950f, 0.99896f);
    turbo_colormap[50] = (float3)(0.24946f, 0.59943f, 0.99835f);
    turbo_colormap[51] = (float3)(0.24427f, 0.60937f, 0.99697f);
    turbo_colormap[52] = (float3)(0.23874f, 0.61931f, 0.99485f);
    turbo_colormap[53] = (float3)(0.23288f, 0.62923f, 0.99202f);
    turbo_colormap[54] = (float3)(0.22676f, 0.63913f, 0.98851f);
    turbo_colormap[55] = (float3)(0.22039f, 0.64901f, 0.98436f);
    turbo_colormap[56] = (float3)(0.21382f, 0.65886f, 0.97959f);
    turbo_colormap[57] = (float3)(0.20708f, 0.66866f, 0.97423f);
    turbo_colormap[58] = (float3)(0.20021f, 0.67842f, 0.96833f);
    turbo_colormap[59] = (float3)(0.19326f, 0.68812f, 0.96190f);
    turbo_colormap[60] = (float3)(0.18625f, 0.69775f, 0.95498f);
    turbo_colormap[61] = (float3)(0.17923f, 0.70732f, 0.94761f);
    turbo_colormap[62] = (float3)(0.17223f, 0.71680f, 0.93981f);
    turbo_colormap[63] = (float3)(0.16529f, 0.72620f, 0.93161f);
    turbo_colormap[64] = (float3)(0.15844f, 0.73551f, 0.92305f);
    turbo_colormap[65] = (float3)(0.15173f, 0.74472f, 0.91416f);
    turbo_colormap[66] = (float3)(0.14519f, 0.75381f, 0.90496f);
    turbo_colormap[67] = (float3)(0.13886f, 0.76279f, 0.89550f);
    turbo_colormap[68] = (float3)(0.13278f, 0.77165f, 0.88580f);
    turbo_colormap[69] = (float3)(0.12698f, 0.78037f, 0.87590f);
    turbo_colormap[70] = (float3)(0.12151f, 0.78896f, 0.86581f);
    turbo_colormap[71] = (float3)(0.11639f, 0.79740f, 0.85559f);
    turbo_colormap[72] = (float3)(0.11167f, 0.80569f, 0.84525f);
    turbo_colormap[73] = (float3)(0.10738f, 0.81381f, 0.83484f);
    turbo_colormap[74] = (float3)(0.10357f, 0.82177f, 0.82437f);
    turbo_colormap[75] = (float3)(0.10026f, 0.82955f, 0.81389f);
    turbo_colormap[76] = (float3)(0.09750f, 0.83714f, 0.80342f);
    turbo_colormap[77] = (float3)(0.09532f, 0.84455f, 0.79299f);
    turbo_colormap[78] = (float3)(0.09377f, 0.85175f, 0.78264f);
    turbo_colormap[79] = (float3)(0.09287f, 0.85875f, 0.77240f);
    turbo_colormap[80] = (float3)(0.09267f, 0.86554f, 0.76230f);
    turbo_colormap[81] = (float3)(0.09320f, 0.87211f, 0.75237f);
    turbo_colormap[82] = (float3)(0.09451f, 0.87844f, 0.74265f);
    turbo_colormap[83] = (float3)(0.09662f, 0.88454f, 0.73316f);
    turbo_colormap[84] = (float3)(0.09958f, 0.89040f, 0.72393f);
    turbo_colormap[85] = (float3)(0.10342f, 0.89600f, 0.71500f);
    turbo_colormap[86] = (float3)(0.10815f, 0.90142f, 0.70599f);
    turbo_colormap[87] = (float3)(0.11374f, 0.90673f, 0.69651f);
    turbo_colormap[88] = (float3)(0.12014f, 0.91193f, 0.68660f);
    turbo_colormap[89] = (float3)(0.12733f, 0.91701f, 0.67627f);
    turbo_colormap[90] = (float3)(0.13526f, 0.92197f, 0.66556f);
    turbo_colormap[91] = (float3)(0.14391f, 0.92680f, 0.65448f);
    turbo_colormap[92] = (float3)(0.15323f, 0.93151f, 0.64308f);
    turbo_colormap[93] = (float3)(0.16319f, 0.93609f, 0.63137f);
    turbo_colormap[94] = (float3)(0.17377f, 0.94053f, 0.61938f);
    turbo_colormap[95] = (float3)(0.18491f, 0.94484f, 0.60713f);
    turbo_colormap[96] = (float3)(0.19659f, 0.94901f, 0.59466f);
    turbo_colormap[97] = (float3)(0.20877f, 0.95304f, 0.58199f);
    turbo_colormap[98] = (float3)(0.22142f, 0.95692f, 0.56914f);
    turbo_colormap[99] = (float3)(0.23449f, 0.96065f, 0.55614f);
    turbo_colormap[100] = (float3)(0.24797f, 0.96423f, 0.54303f);
    turbo_colormap[101] = (float3)(0.26180f, 0.96765f, 0.52981f);
    turbo_colormap[102] = (float3)(0.27597f, 0.97092f, 0.51653f);
    turbo_colormap[103] = (float3)(0.29042f, 0.97403f, 0.50321f);
    turbo_colormap[104] = (float3)(0.30513f, 0.97697f, 0.48987f);
    turbo_colormap[105] = (float3)(0.32006f, 0.97974f, 0.47654f);
    turbo_colormap[106] = (float3)(0.33517f, 0.98234f, 0.46325f);
    turbo_colormap[107] = (float3)(0.35043f, 0.98477f, 0.45002f);
    turbo_colormap[108] = (float3)(0.36581f, 0.98702f, 0.43688f);
    turbo_colormap[109] = (float3)(0.38127f, 0.98909f, 0.42386f);
    turbo_colormap[110] = (float3)(0.39678f, 0.99098f, 0.41098f);
    turbo_colormap[111] = (float3)(0.41229f, 0.99268f, 0.39826f);
    turbo_colormap[112] = (float3)(0.42778f, 0.99419f, 0.38575f);
    turbo_colormap[113] = (float3)(0.44321f, 0.99551f, 0.37345f);
    turbo_colormap[114] = (float3)(0.45854f, 0.99663f, 0.36140f);
    turbo_colormap[115] = (float3)(0.47375f, 0.99755f, 0.34963f);
    turbo_colormap[116] = (float3)(0.48879f, 0.99828f, 0.33816f);
    turbo_colormap[117] = (float3)(0.50362f, 0.99879f, 0.32701f);
    turbo_colormap[118] = (float3)(0.51822f, 0.99910f, 0.31622f);
    turbo_colormap[119] = (float3)(0.53255f, 0.99919f, 0.30581f);
    turbo_colormap[120] = (float3)(0.54658f, 0.99907f, 0.29581f);
    turbo_colormap[121] = (float3)(0.56026f, 0.99873f, 0.28623f);
    turbo_colormap[122] = (float3)(0.57357f, 0.99817f, 0.27712f);
    turbo_colormap[123] = (float3)(0.58646f, 0.99739f, 0.26849f);
    turbo_colormap[124] = (float3)(0.59891f, 0.99638f, 0.26038f);
    turbo_colormap[125] = (float3)(0.61088f, 0.99514f, 0.25280f);
    turbo_colormap[126] = (float3)(0.62233f, 0.99366f, 0.24579f);
    turbo_colormap[127] = (float3)(0.63323f, 0.99195f, 0.23937f);
    turbo_colormap[128] = (float3)(0.64362f ,0.98999f, 0.23356f);
    turbo_colormap[129] = (float3)(0.65394f, 0.98775f, 0.22835f);
    turbo_colormap[130] = (float3)(0.66428f, 0.98524f, 0.22370f);
    turbo_colormap[131] = (float3)(0.67462f, 0.98246f, 0.21960f);
    turbo_colormap[132] = (float3)(0.68494f, 0.97941f, 0.21602f);
    turbo_colormap[133] = (float3)(0.69525f, 0.97610f, 0.21294f);
    turbo_colormap[134] = (float3)(0.70553f, 0.97255f, 0.21032f);
    turbo_colormap[135] = (float3)(0.71577f, 0.96875f, 0.20815f);
    turbo_colormap[136] = (float3)(0.72596f, 0.96470f, 0.20640f);
    turbo_colormap[137] = (float3)(0.73610f, 0.96043f, 0.20504f);
    turbo_colormap[138] = (float3)(0.74617f, 0.95593f, 0.20406f);
    turbo_colormap[139] = (float3)(0.75617f, 0.95121f, 0.20343f);
    turbo_colormap[140] = (float3)(0.76608f, 0.94627f, 0.20311f);
    turbo_colormap[141] = (float3)(0.77591f, 0.94113f, 0.20310f);
    turbo_colormap[142] = (float3)(0.78563f, 0.93579f, 0.20336f);
    turbo_colormap[143] = (float3)(0.79524f, 0.93025f, 0.20386f);
    turbo_colormap[144] = (float3)(0.80473f, 0.92452f, 0.20459f);
    turbo_colormap[145] = (float3)(0.81410f, 0.91861f, 0.20552f);
    turbo_colormap[146] = (float3)(0.82333f, 0.91253f, 0.20663f);
    turbo_colormap[147] = (float3)(0.83241f, 0.90627f, 0.20788f);
    turbo_colormap[148] = (float3)(0.84133f, 0.89986f, 0.20926f);
    turbo_colormap[149] = (float3)(0.85010f, 0.89328f, 0.21074f);
    turbo_colormap[150] = (float3)(0.85868f, 0.88655f, 0.21230f);
    turbo_colormap[151] = (float3)(0.86709f, 0.87968f, 0.21391f);
    turbo_colormap[152] = (float3)(0.87530f, 0.87267f, 0.21555f);
    turbo_colormap[153] = (float3)(0.88331f, 0.86553f, 0.21719f);
    turbo_colormap[154] = (float3)(0.89112f, 0.85826f, 0.21880f);
    turbo_colormap[155] = (float3)(0.89870f, 0.85087f, 0.22038f);
    turbo_colormap[156] = (float3)(0.90605f, 0.84337f, 0.22188f);
    turbo_colormap[157] = (float3)(0.91317f, 0.83576f, 0.22328f);
    turbo_colormap[158] = (float3)(0.92004f, 0.82806f, 0.22456f);
    turbo_colormap[159] = (float3)(0.92666f, 0.82025f, 0.22570f);
    turbo_colormap[160] = (float3)(0.93301f, 0.81236f, 0.22667f);
    turbo_colormap[161] = (float3)(0.93909f, 0.80439f, 0.22744f);
    turbo_colormap[162] = (float3)(0.94489f, 0.79634f, 0.22800f);
    turbo_colormap[163] = (float3)(0.95039f, 0.78823f, 0.22831f);
    turbo_colormap[164] = (float3)(0.95560f, 0.78005f, 0.22836f);
    turbo_colormap[165] = (float3)(0.96049f, 0.77181f, 0.22811f);
    turbo_colormap[166] = (float3)(0.96507f, 0.76352f, 0.22754f);
    turbo_colormap[167] = (float3)(0.96931f, 0.75519f, 0.22663f);
    turbo_colormap[168] = (float3)(0.97323f, 0.74682f, 0.22536f);
    turbo_colormap[169] = (float3)(0.97679f, 0.73842f, 0.22369f);
    turbo_colormap[170] = (float3)(0.98000f, 0.73000f, 0.22161f);
    turbo_colormap[171] = (float3)(0.98289f, 0.72140f, 0.21918f);
    turbo_colormap[172] = (float3)(0.98549f, 0.71250f, 0.21650f);
    turbo_colormap[173] = (float3)(0.98781f, 0.70330f, 0.21358f);
    turbo_colormap[174] = (float3)(0.98986f, 0.69382f, 0.21043f);
    turbo_colormap[175] = (float3)(0.99163f, 0.68408f, 0.20706f);
    turbo_colormap[176] = (float3)(0.99314f, 0.67408f, 0.20348f);
    turbo_colormap[177] = (float3)(0.99438f, 0.66386f, 0.19971f);
    turbo_colormap[178] = (float3)(0.99535f, 0.65341f, 0.19577f);
    turbo_colormap[179] = (float3)(0.99607f, 0.64277f, 0.19165f);
    turbo_colormap[180] = (float3)(0.99654f, 0.63193f, 0.18738f);
    turbo_colormap[181] = (float3)(0.99675f, 0.62093f, 0.18297f);
    turbo_colormap[182] = (float3)(0.99672f, 0.60977f, 0.17842f);
    turbo_colormap[183] = (float3)(0.99644f, 0.59846f, 0.17376f);
    turbo_colormap[184] = (float3)(0.99593f, 0.58703f, 0.16899f);
    turbo_colormap[185] = (float3)(0.99517f, 0.57549f, 0.16412f);
    turbo_colormap[186] = (float3)(0.99419f, 0.56386f, 0.15918f);
    turbo_colormap[187] = (float3)(0.99297f, 0.55214f, 0.15417f);
    turbo_colormap[188] = (float3)(0.99153f, 0.54036f, 0.14910f);
    turbo_colormap[189] = (float3)(0.98987f, 0.52854f, 0.14398f);
    turbo_colormap[190] = (float3)(0.98799f, 0.51667f, 0.13883f);
    turbo_colormap[191] = (float3)(0.98590f, 0.50479f, 0.13367f);
    turbo_colormap[192] = (float3)(0.98360f, 0.49291f, 0.12849f);
    turbo_colormap[193] = (float3)(0.98108f, 0.48104f, 0.12332f);
    turbo_colormap[194] = (float3)(0.97837f, 0.46920f, 0.11817f);
    turbo_colormap[195] = (float3)(0.97545f, 0.45740f, 0.11305f);
    turbo_colormap[196] = (float3)(0.97234f, 0.44565f, 0.10797f);
    turbo_colormap[197] = (float3)(0.96904f, 0.43399f, 0.10294f);
    turbo_colormap[198] = (float3)(0.96555f, 0.42241f, 0.09798f);
    turbo_colormap[199] = (float3)(0.96187f, 0.41093f, 0.09310f);
    turbo_colormap[200] = (float3)(0.95801f, 0.39958f, 0.08831f);
    turbo_colormap[201] = (float3)(0.95398f, 0.38836f, 0.08362f);
    turbo_colormap[202] = (float3)(0.94977f, 0.37729f, 0.07905f);
    turbo_colormap[203] = (float3)(0.94538f, 0.36638f, 0.07461f);
    turbo_colormap[204] = (float3)(0.94084f, 0.35566f, 0.07031f);
    turbo_colormap[205] = (float3)(0.93612f, 0.34513f, 0.06616f);
    turbo_colormap[206] = (float3)(0.93125f, 0.33482f, 0.06218f);
    turbo_colormap[207] = (float3)(0.92623f, 0.32473f, 0.05837f);
    turbo_colormap[208] = (float3)(0.92105f, 0.31489f, 0.05475f);
    turbo_colormap[209] = (float3)(0.91572f, 0.30530f, 0.05134f);
    turbo_colormap[210] = (float3)(0.91024f, 0.29599f, 0.04814f);
    turbo_colormap[211] = (float3)(0.90463f, 0.28696f, 0.04516f);
    turbo_colormap[212] = (float3)(0.89888f, 0.27824f, 0.04243f);
    turbo_colormap[213] = (float3)(0.89298f, 0.26981f, 0.03993f);
    turbo_colormap[214] = (float3)(0.88691f, 0.26152f, 0.03753f);
    turbo_colormap[215] = (float3)(0.88066f, 0.25334f, 0.03521f);
    turbo_colormap[216] = (float3)(0.87422f, 0.24526f, 0.03297f);
    turbo_colormap[217] = (float3)(0.86760f, 0.23730f, 0.03082f);
    turbo_colormap[218] = (float3)(0.86079f, 0.22945f, 0.02875f);
    turbo_colormap[219] = (float3)(0.85380f, 0.22170f, 0.02677f);
    turbo_colormap[220] = (float3)(0.84662f, 0.21407f, 0.02487f);
    turbo_colormap[221] = (float3)(0.83926f, 0.20654f, 0.02305f);
    turbo_colormap[222] = (float3)(0.83172f, 0.19912f, 0.02131f);
    turbo_colormap[223] = (float3)(0.82399f, 0.19182f, 0.01966f);
    turbo_colormap[224] = (float3)(0.81608f, 0.18462f, 0.01809f);
    turbo_colormap[225] = (float3)(0.80799f, 0.17753f, 0.01660f);
    turbo_colormap[226] = (float3)(0.79971f, 0.17055f, 0.01520f);
    turbo_colormap[227] = (float3)(0.79125f, 0.16368f, 0.01387f);
    turbo_colormap[228] = (float3)(0.78260f, 0.15693f, 0.01264f);
    turbo_colormap[229] = (float3)(0.77377f, 0.15028f, 0.01148f);
    turbo_colormap[230] = (float3)(0.76476f, 0.14374f, 0.01041f);
    turbo_colormap[231] = (float3)(0.75556f, 0.13731f, 0.00942f);
    turbo_colormap[232] = (float3)(0.74617f, 0.13098f, 0.00851f);
    turbo_colormap[233] = (float3)(0.73661f, 0.12477f, 0.00769f);
    turbo_colormap[234] = (float3)(0.72686f, 0.11867f, 0.00695f);
    turbo_colormap[235] = (float3)(0.71692f, 0.11268f, 0.00629f);
    turbo_colormap[236] = (float3)(0.70680f, 0.10680f, 0.00571f);
    turbo_colormap[237] = (float3)(0.69650f, 0.10102f, 0.00522f);
    turbo_colormap[238] = (float3)(0.68602f, 0.09536f, 0.00481f);
    turbo_colormap[239] = (float3)(0.67535f, 0.08980f, 0.00449f);
    turbo_colormap[240] = (float3)(0.66449f, 0.08436f, 0.00424f);
    turbo_colormap[241] = (float3)(0.65345f, 0.07902f, 0.00408f);
    turbo_colormap[242] = (float3)(0.64223f, 0.07380f, 0.00401f);
    turbo_colormap[243] = (float3)(0.63082f, 0.06868f, 0.00401f);
    turbo_colormap[244] = (float3)(0.61923f, 0.06367f, 0.00410f);
    turbo_colormap[245] = (float3)(0.60746f, 0.05878f, 0.00427f);
    turbo_colormap[246] = (float3)(0.59550f, 0.05399f, 0.00453f);
    turbo_colormap[247] = (float3)(0.58336f, 0.04931f, 0.00486f);
    turbo_colormap[248] = (float3)(0.57103f, 0.04474f, 0.00529f);
    turbo_colormap[249] = (float3)(0.55852f, 0.04028f, 0.00579f);
    turbo_colormap[250] = (float3)(0.54583f, 0.03593f, 0.00638f);
    turbo_colormap[251] = (float3)(0.53295f, 0.03169f, 0.00705f);
    turbo_colormap[252] = (float3)(0.51989f, 0.02756f, 0.00780f);
    turbo_colormap[253] = (float3)(0.50664f, 0.02354f, 0.00863f);
    turbo_colormap[254] = (float3)(0.49321f, 0.01963f, 0.00955f);
    turbo_colormap[255] = (float3)(0.47960f, 0.01583f, 0.01055f);

    i = round(255*intensity);

    // Clamping low values:
    if(i < 0)
    {
        i = 0;
    }

    // Clamping high values:
    if(i > 255)
    {
        i = 255;
    }

    return (float3)(turbo_colormap[i]);
}
