#include <SDL2/SDL.h>
#include <mpfr.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 800
#define HEIGHT 600
// #define WIDTH 1920
// #define HEIGHT 1080
#define MAX_ITER 5000
#define ESCAPE_RADIUS 4

#define FP 1024

void
render (uint32_t *pixels, mpfr_t center_re_mpfr, mpfr_t center_im_mpfr,
        double scale)
{
  double z_ref_re[MAX_ITER];
  double z_ref_im[MAX_ITER];

  mpfr_t z_re, z_im, temp_re, temp_im, re_sqr, im_sqr, escape_radius_sqr;
  mpfr_inits2 (FP, z_re, z_im, temp_re, temp_im, re_sqr, im_sqr,
               escape_radius_sqr, (mpfr_ptr)0);

  mpfr_set_d (z_re, 0.0, MPFR_RNDN);
  mpfr_set_d (z_im, 0.0, MPFR_RNDN);

  mpfr_set_d (escape_radius_sqr, ESCAPE_RADIUS * ESCAPE_RADIUS, MPFR_RNDN);
  mpfr_mul (escape_radius_sqr, escape_radius_sqr, escape_radius_sqr, MPFR_RNDN);

  int ref_iter = 0;

  const Uint32 time_start_orbit = SDL_GetTicks ();

  while (ref_iter < MAX_ITER)
    {
      z_ref_re[ref_iter] = mpfr_get_d (z_re, MPFR_RNDN);
      z_ref_im[ref_iter] = mpfr_get_d (z_im, MPFR_RNDN);

      mpfr_mul (re_sqr, z_re, z_re, MPFR_RNDN);
      mpfr_mul (im_sqr, z_im, z_im, MPFR_RNDN);

      mpfr_sub (temp_re, re_sqr, im_sqr, MPFR_RNDN);

      mpfr_mul (temp_im, z_re, z_im, MPFR_RNDN);
      mpfr_mul_ui (temp_im, temp_im, 2, MPFR_RNDN);

      mpfr_add (z_re, temp_re, center_re_mpfr, MPFR_RNDN);
      mpfr_add (z_im, temp_im, center_im_mpfr, MPFR_RNDN);

      mpfr_mul (re_sqr, z_re, z_re, MPFR_RNDN);
      mpfr_mul (im_sqr, z_im, z_im, MPFR_RNDN);
      mpfr_add (temp_re, re_sqr, im_sqr, MPFR_RNDN);

      if (mpfr_greater_p (temp_re, escape_radius_sqr))
        break;

      ref_iter++;
    }

  const Uint32 time_end_orbit = SDL_GetTicks ();
  const Uint32 time_start_perturbation = SDL_GetTicks ();

  uint32_t palette[] = {
    0xFF000000,
    0xFF1A0A5E,
    0xFF3D1F99,
    0xFF5C44C3,
    0xFF7C68E5,
    0xFF9AA1F1,
    0xFFB7BCFA,
    0xFFDFE5FF,
    0xFFB1C1D9,
    0xFF7D91BF,
    0xFF4C65A7,
    0xFF1F3D88,
    0xFF0A1A5E,
  };
  int palette_size = sizeof (palette) / sizeof (palette[0]);

#pragma omp parallel for schedule(dynamic, 1)
  for (int y = 0; y < HEIGHT; y++)
    {
      for (int x = 0; x < WIDTH; x++)
        {
          double dx = (x - WIDTH / 2.0) * scale;
          double dy = (y - HEIGHT / 2.0) * scale;

          double delta_c_re = dx;
          double delta_c_im = dy;
          double delta_z_re = 0.0;
          double delta_z_im = 0.0;

          int iter = 0;
          int ref_i = 0;

          while (iter < MAX_ITER)
            {
              if (ref_i >= ref_iter)
                {
                  delta_z_re = z_ref_re[ref_i - 1] + delta_z_re;
                  delta_z_im = z_ref_im[ref_i - 1] + delta_z_im;
                  ref_i = 0;
                }

              double ref_re = z_ref_re[ref_i];
              double ref_im = z_ref_im[ref_i];

              double temp_re
                  = 2.0 * (ref_re * delta_z_re - ref_im * delta_z_im);
              double temp_im
                  = 2.0 * (ref_re * delta_z_im + ref_im * delta_z_re);

              double dz2_re = delta_z_re * delta_z_re - delta_z_im * delta_z_im;
              double dz2_im = 2.0 * delta_z_re * delta_z_im;

              delta_z_re = temp_re + dz2_re + delta_c_re;
              delta_z_im = temp_im + dz2_im + delta_c_im;

              ref_i++;

              double z_re = z_ref_re[ref_i] + delta_z_re;
              double z_im = z_ref_im[ref_i] + delta_z_im;

              if (z_re * z_re + z_im * z_im > ESCAPE_RADIUS)
                break;

              if ((delta_z_re * delta_z_re + delta_z_im * delta_z_im)
                  > (z_re * z_re + z_im * z_im))
                {
                  delta_z_re = z_re;
                  delta_z_im = z_im;
                  ref_i = 0;
                }

              iter++;
            }

          if (iter == MAX_ITER)
            {
              pixels[y * WIDTH + x] = 0xFF000000;
            }
          else
            {
              float t = (float)(iter % palette_size)
                        + ((float)(iter % 1));


              float freq = 0.1f;
              t = (float)(iter * freq);
              t = t
                  - floorf (t / palette_size)
                        * palette_size;

              int idx = (int)t;
              float frac = t - idx;

              uint32_t c1 = palette[idx % palette_size];
              uint32_t c2 = palette[(idx + 1) % palette_size];

              uint8_t r = (uint8_t)(((1 - frac) * ((c1 >> 16) & 0xFF))
                                    + (frac * ((c2 >> 16) & 0xFF)));
              uint8_t g = (uint8_t)(((1 - frac) * ((c1 >> 8) & 0xFF))
                                    + (frac * ((c2 >> 8) & 0xFF)));
              uint8_t b = (uint8_t)(((1 - frac) * (c1 & 0xFF))
                                    + (frac * (c2 & 0xFF)));

              pixels[y * WIDTH + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
    }

  const Uint32 time_end_perturbation = SDL_GetTicks ();

  printf ("render:         %dms\n",
          (time_end_orbit - time_start_orbit)
              + (time_end_perturbation - time_start_perturbation));
  printf ("  orbit:        %dms\n", time_end_orbit - time_start_orbit);
  printf ("  perturbation: %dms\n",
          time_end_perturbation - time_start_perturbation);

  mpfr_clears (z_re, z_im, temp_re, temp_im, re_sqr, im_sqr, escape_radius_sqr,
               (mpfr_ptr)0);
}

int
main (int argc, char *argv[])
{
  if (SDL_Init (SDL_INIT_VIDEO) != 0)
    {
      fprintf (stderr, "SDL_Init error: %s\n", SDL_GetError ());
      return 1;
    }

  SDL_Window *window = SDL_CreateWindow (
      "Mandelbrot Perturbation MPFR", SDL_WINDOWPOS_CENTERED,
      SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
  if (!window)
    {
      fprintf (stderr, "SDL_CreateWindow error: %s\n", SDL_GetError ());
      SDL_Quit ();
      return 1;
    }

  SDL_Renderer *renderer
      = SDL_CreateRenderer (window, -1, SDL_RENDERER_ACCELERATED);
  if (!renderer)
    {
      fprintf (stderr, "SDL_CreateRenderer error: %s\n", SDL_GetError ());
      SDL_DestroyWindow (window);
      SDL_Quit ();
      return 1;
    }

  SDL_Texture *texture
      = SDL_CreateTexture (renderer, SDL_PIXELFORMAT_ARGB8888,
                           SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
  if (!texture)
    {
      fprintf (stderr, "SDL_CreateTexture error: %s\n", SDL_GetError ());
      SDL_DestroyRenderer (renderer);
      SDL_DestroyWindow (window);
      SDL_Quit ();
      return 1;
    }

  uint32_t *pixels = malloc (WIDTH * HEIGHT * sizeof (uint32_t));
  if (!pixels)
    {
      fprintf (stderr, "malloc failed\n");
      SDL_DestroyTexture (texture);
      SDL_DestroyRenderer (renderer);
      SDL_DestroyWindow (window);
      SDL_Quit ();
      return 1;
    }

  mpfr_t center_re_mpfr, center_im_mpfr, scale_mpfr;
  mpfr_inits2 (FP, center_re_mpfr, center_im_mpfr, scale_mpfr, (mpfr_ptr)0);

  mpfr_set_str (center_re_mpfr, "0", 10,
                MPFR_RNDN);
  mpfr_set_str (center_im_mpfr, "0", 10, MPFR_RNDN);

  mpfr_set_d (scale_mpfr, 0.005, MPFR_RNDN);

  double center_re_d = mpfr_get_d (center_re_mpfr, MPFR_RNDN);
  double center_im_d = mpfr_get_d (center_im_mpfr, MPFR_RNDN);
  double scale_d = mpfr_get_d (scale_mpfr, MPFR_RNDN);

  int redraw = 1;
  int quit = 0;
  SDL_Event e;

  const Uint8 *keys;
  double moveSpeed = 0.1;

  double zoom_x = 0.0;
  double zoom_y = 0.0;
  double zoom_value = 1.0;
  int zoom_expect_recompute = 0;
  int zoom_last_time = 0;

  while (!quit)
    {
      keys = SDL_GetKeyboardState (NULL);

      double pan = moveSpeed * scale_d * 50;

      if (keys[SDL_SCANCODE_W])
        {
          mpfr_sub_d (center_im_mpfr, center_im_mpfr, pan, MPFR_RNDN);
          redraw = 1;
        }
      if (keys[SDL_SCANCODE_S])
        {
          mpfr_add_d (center_im_mpfr, center_im_mpfr, pan, MPFR_RNDN);
          redraw = 1;
        }
      if (keys[SDL_SCANCODE_A])
        {
          mpfr_sub_d (center_re_mpfr, center_re_mpfr, pan, MPFR_RNDN);
          redraw = 1;
        }
      if (keys[SDL_SCANCODE_D])
        {
          mpfr_add_d (center_re_mpfr, center_re_mpfr, pan, MPFR_RNDN);
          redraw = 1;
        }

      if (keys[SDL_SCANCODE_S])
        {
          mpfr_mul_d (scale_mpfr, scale_mpfr, 0.9, MPFR_RNDN);
          redraw = 1;
        }

      while (SDL_PollEvent (&e))
        {
          if (e.type == SDL_QUIT)
            quit = 1;
          else if (e.type == SDL_MOUSEWHEEL)
            {
              int mx, my;
              SDL_GetMouseState (&mx, &my);

              if (zoom_value == 1.0)
                {
                  zoom_x = mx;
                  zoom_y = my;
                }

              zoom_value *= e.wheel.y > 0 ? 1.25 : 0.75;

              zoom_expect_recompute = 1;
              zoom_last_time = SDL_GetTicks ();
            }
        }

      if (zoom_expect_recompute)
        {
          int time = SDL_GetTicks ();

          if (time - zoom_last_time > 300)
            {
              mpfr_t old_scale_mpfr, new_scale_mpfr;
              mpfr_t re_before, im_before;
              mpfr_t tmp1, tmp2;

              mpfr_inits2 (FP, old_scale_mpfr, new_scale_mpfr, re_before,
                           im_before, tmp1, tmp2, (mpfr_ptr)0);

              mpfr_set (old_scale_mpfr, scale_mpfr, MPFR_RNDN);

              mpfr_set_d (tmp1, zoom_value, MPFR_RNDN);
              mpfr_div (new_scale_mpfr, scale_mpfr, tmp1, MPFR_RNDN);

              mpfr_set_d (tmp1, (double)(zoom_x - WIDTH / 2.0), MPFR_RNDN);
              mpfr_mul (tmp2, tmp1, old_scale_mpfr, MPFR_RNDN);
              mpfr_add (re_before, center_re_mpfr, tmp2, MPFR_RNDN);

              mpfr_set_d (tmp1, (double)(zoom_y - HEIGHT / 2.0), MPFR_RNDN);
              mpfr_mul (tmp2, tmp1, old_scale_mpfr, MPFR_RNDN);
              mpfr_add (im_before, center_im_mpfr, tmp2, MPFR_RNDN);

              mpfr_set_d (tmp1, (double)(zoom_x - WIDTH / 2.0), MPFR_RNDN);
              mpfr_mul (tmp2, tmp1, new_scale_mpfr, MPFR_RNDN);
              mpfr_sub (center_re_mpfr, re_before, tmp2, MPFR_RNDN);

              mpfr_set_d (tmp1, (double)(zoom_y - HEIGHT / 2.0), MPFR_RNDN);
              mpfr_mul (tmp2, tmp1, new_scale_mpfr, MPFR_RNDN);
              mpfr_sub (center_im_mpfr, im_before, tmp2, MPFR_RNDN);

              mpfr_div_d (scale_mpfr, scale_mpfr, zoom_value, MPFR_RNDN);

              scale_d = mpfr_get_d (scale_mpfr, MPFR_RNDN);

              redraw = 1;

              zoom_value = 1.0;
              zoom_expect_recompute = 0;

              mpfr_clears (old_scale_mpfr, new_scale_mpfr, re_before, im_before,
                           tmp1, tmp2, (mpfr_ptr)0);
            }
        }

      if (redraw)
        {
          center_re_d = mpfr_get_d (center_re_mpfr, MPFR_RNDN);
          center_im_d = mpfr_get_d (center_im_mpfr, MPFR_RNDN);
          scale_d = mpfr_get_d (scale_mpfr, MPFR_RNDN);

          render (pixels, center_re_mpfr, center_im_mpfr, scale_d);

          SDL_UpdateTexture (texture, NULL, pixels, WIDTH * sizeof (uint32_t));
          SDL_RenderClear (renderer);

          printf ("Z=%.2e\n", scale_d);
          redraw = 0;
        }

      SDL_Rect vr = {
        .x = zoom_x - zoom_x * zoom_value,
        .y = zoom_y - zoom_y * zoom_value,
        .w = WIDTH * zoom_value,
        .h = HEIGHT * zoom_value,
      };

      SDL_RenderCopy (renderer, texture, NULL, &vr);
      SDL_RenderPresent (renderer);

      SDL_Delay (16);
    }

  free (pixels);
  SDL_DestroyTexture (texture);
  SDL_DestroyRenderer (renderer);
  SDL_DestroyWindow (window);

  mpfr_clears (center_re_mpfr, center_im_mpfr, scale_mpfr, (mpfr_ptr)0);
  SDL_Quit ();

  return 0;
}

