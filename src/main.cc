#include "jpeg_codec.h"

int main(int argc, char** argv) {
  gfx::JPEGCodec::Encode(NULL, gfx::JPEGCodec::FORMAT_RGB, 0, 0, 0, 0, NULL);
  return 0;
}
