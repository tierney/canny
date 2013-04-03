#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT([Canny], [1.0], [tierney@cs.nyu.edu])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE

AC_CANONICAL_HOST

AM_SILENT_RULES([yes])

AC_CONFIG_SRCDIR([src/jpeg_codec.cc])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])
LT_PREREQ([2.2])
LT_INIT([dlopen])

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AM_PROG_CC_C_O
AC_PROG_GREP
AC_PROG_EGREP
AC_PROG_CXX

dnl First look for Boost, maybe for a specific minimum version:
dnl BOOST_REQUIRE([1.34])

dnl Then look for specific Boost libraries you need:
dnl BOOST_ARRAY
dnl BOOST_DATE_TIME


dnl # TODO(chandlerc@google.com): Currently we aren't running the Python tests
dnl # against the interpreter detected by AM_PATH_PYTHON, and so we condition
dnl # HAVE_PYTHON by requiring "python" to be in the PATH, and that interpreter's
dnl # version to be >= 2.3. This will allow the scripts to use a "/usr/bin/env"
dnl # hashbang.
dnl PYTHON=  # We *do not* allow the user to specify a python interpreter
dnl AC_PATH_PROG([PYTHON],[python],[:])
dnl AS_IF([test "$PYTHON" != ":"],
dnl       [AM_PYTHON_CHECK_VERSION([$PYTHON],[2.3],[:],[PYTHON=":"])])
dnl AM_CONDITIONAL([HAVE_PYTHON],[test "$PYTHON" != ":"])

dnl # Configure pthreads.
dnl AC_ARG_WITH([pthreads],
dnl             [AS_HELP_STRING([--with-pthreads],
dnl                [use pthreads (default is yes)])],
dnl             [with_pthreads=$withval],
dnl             [with_pthreads=check])

dnl have_pthreads=no
dnl AS_IF([test "x$with_pthreads" != "xno"],
dnl       [ACX_PTHREAD(
dnl         [],
dnl         [AS_IF([test "x$with_pthreads" != "xcheck"],
dnl                [AC_MSG_FAILURE(
dnl                  [--with-pthreads was specified, but unable to be used])])])
dnl        have_pthreads="$acx_pthread_ok"])
dnl AM_CONDITIONAL([HAVE_PTHREADS],[test "x$have_pthreads" == "xyes"])

AC_SUBST(PTHREAD_CFLAGS)
AC_SUBST(PTHREAD_LIBS)

# Checks for libraries.
AC_CHECK_LIB([glog], [_ZN6google17InitGoogleLoggingEPKc], ,
  AC_MSG_WARN([

Make sure you link to google's logging library.

]))

AC_CHECK_LIB([jpeg], [jpeg_finish_compress], ,
  AC_MSG_WARN([

Make sure you link to libjpeg.

]))

dnl libtcmalloc

dnl AC_ARG_WITH([tcmalloc],
dnl   [AS_HELP_STRING([--with-tcmalloc],
dnl     [enable support for tcmalloc])],
dnl   [with_tcmalloc=yes],
dnl   [with_tcmalloc=no])

dnl AS_IF([test "x$with_tcmalloc" != xno],
dnl   [AC_CHECK_LIB([tcmalloc], [tc_malloc],
dnl     [AC_SUBST([LIBTCMALLOC], ["-lprofiler -ltcmalloc"])
dnl      AC_DEFINE([HAVE_LIBTCMALLOC], [1],
dnl                [Define if you have libtcmalloc])
dnl     ],
dnl     [AC_MSG_FAILURE(
dnl        [tcmalloc test failed (--without-tcmalloc to disable)])],
dnl     [])])
dnl AM_CONDITIONAL(USE_LIBTCMALLOC, test "x$with_tcmalloc" != "xno")

dnl Boost specific suffices.
dnl AS_IF([test -z "${BOOST_LIB_SUFFIX+x}"], [BOOST_LIB_SUFFIX=-mt])
dnl AC_ARG_VAR([BOOST_LIB_SUFFIX], [Boost library name suffix [default=-mt]])

BOOST_SYSTEM

dnl AC_CHECK_LIB([gtest], [_ZN7testing14InitGoogleTestEPiPPc], ,
dnl   AC_MSG_WARN([

dnl Missing gtest library. (TODO(tierney): Figure out a correct function name.)
dnl ]))

dnl AC_CHECK_LIB([gtest_main], [main], ,
dnl   AC_MSG_ERROR([
dnl Missing gtest_main.
dnl Fetch from: http://code.google.com/p/googletest
dnl ]))

# Checks for header files.
AC_CHECK_HEADERS([limits.h ltdl.h stddef.h stdint.h string.h jpeglib.h])

AC_SEARCH_LIBS([lt_dlopen], [ltdl])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_UINT32_T
AC_TYPE_UINT8_T
AC_CHECK_TYPES([ptrdiff_t])

# Checks for library functions.
AC_CHECK_FUNCS([bzero memmove])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 ])

# AC_CONFIG_FILES([scripts/gtest-config], [chmod +x scripts/gtest-config])
AC_OUTPUT
