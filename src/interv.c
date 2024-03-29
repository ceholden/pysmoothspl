/*
 *  R : A Computer Language for Statistical Data Analysis
 *  Copyright (C) 2002	      The R Core Team
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  http://www.r-project.org/Licenses/
 *
 * The interv() used to be Fortran in ../library/modreg/src/bvalue.f
 * and part of Hastie and Tibshirani's public domain GAMFIT package.
 *
 * Translated by f2c (version 20010821), cleaned up and extended by
 * Martin Maechler.
 */

#include <stdio.h>

/*
int interv_(double *xt, int *n, double *x,
		    int *rightmost_closed, int *all_inside,
		    int *ilo, int *mflag)
{
  return newFindInterval(xt, *n, *x, *rightmost_closed, *all_inside, *ilo, mflag);
}
*/
int findInterval(double *xt, int n, double x,
		  int rightmost_closed,  int all_inside, int ilo,
		  int *mflag)
{
    int istep, middle, ihi;

/* computes  `left' := max( i ; 1 <= i <= n   &&  xt[i] <= x )  .

 ******  i n p u t  ******

  xt	numeric vector of length  n , assumed to be nondecreasing
  n	length(xt)
  x	the point whose location with respect to the sequence  xt  is
        to be determined.
  rightmost_closed {logical} indicating if the rightmost xt[] interval
  	should be closed, i.e. result := n-1 if x == x[n]
  mflag =: all_inside	    {logical} indicating if result should be coerced
		to lie inside {1, n-1}
  ilo   typically the result of the last call to findInterval(.)
        `ilo' used to be a static variable (in Fortran) which is not
	desirable in R anymore (threads!).
	Instead, you *should* use a reasonable value, in the first call.

 ******  o u t p u t  ******

  left, mflag  both integers, whose value is

   0     -1      if            x <  xt[1]
   i      0      if  xt[i]  <= x <  xt[i+1]
   n      1      if  xt[n]  <= x

        in particular,  mflag = 0 is the 'usual' case.  mflag != 0
        indicates that  x  lies outside the halfopen interval
        xt[1] <= y < xt[n] . the asymmetric treatment of the
        interval is due to the decision to make all pp functions cont-
        inuous from the right.

   Note that if all_inside, left is 1 instead of 0 and n-1 instead of n;
   and if rightmost_closed and x == xt[n],  left is    n-1 instead of n.


 ******  m e t h o d  ******

  the program is designed to be efficient in the common situation that
  it is called repeatedly, with  x  taken from an increasing or decreasing
  sequence. this will happen, e.g., when a pp function is to be graphed.
  The first guess for  left  is therefore taken to be the value returned at
  the previous call and stored in the  l o c a l   variable  ilo .
  a first check ascertains that  ilo < n (this is necessary since the
  present call may have nothing to do with the previous call).

  then, if  xt[ilo] <= x < xt[ilo+1], we set  left = ilo
  and are done after just three comparisons.
  otherwise, we repeatedly double the difference  istep = ihi - ilo
  while also moving  ilo  and  ihi  in the direction of  x , until
                      xt[ilo] <= x < xt[ihi] ,
  after which we use bisection to get, in addition, ilo+1 = ihi .
  left = ilo  is then returned.
*/

#define left_boundary  { *mflag = -1;	return(all_inside ? 1 : 0); }

#define right_boundary { *mflag = +1;					\
	return((all_inside || (rightmost_closed && x == xt[n]))		\
		? (n - 1) : n); }


    /* 1-indexing : */
    --xt;

    if(ilo <= 0) {
	if (x < xt[1])			left_boundary;
	ilo = 1;
    }
    ihi = ilo + 1;
    if (ihi >= n) {
	if (x >= xt[n])			right_boundary;
	if (n <= 1) /* x < xt[1] */	left_boundary;
	ilo = n - 1;
	ihi = n;
    }

    if (x < xt[ihi]) {
	if (x >= xt[ilo]) { /* `lucky': same interval as last time */
	    *mflag = 0;	   return ilo;
	}
	/* **** now x < xt[ilo] .	decrease  ilo  to capture  x */
	for(istep = 1; ; istep *= 2) {
	    ihi = ilo;
	    ilo = ihi - istep;
	    if (ilo <= 1)
		break;
	    if (x >= xt[ilo])		goto L50;
	}
	ilo = 1;
	if (x < xt[1])			left_boundary;
    }
    else {
	/* **** now x >= xt[ihi] .	increase  ihi  to capture  x */
	for(istep = 1; ; istep *= 2) {
	    ilo = ihi;
	    ihi = ilo + istep;
	    if (ihi >= n)
		break;
	    if (x < xt[ihi])		goto L50;
	}
	if (x >= xt[n])			right_boundary;
	ihi = n;
    }

  L50:
    /* **** now xt[ilo] <= x < xt[ihi] . narrow the interval. */
    for(;;) {
	middle = (ilo + ihi) / 2;
	if (middle == ilo) {
	    *mflag = 0;	   return ilo;
	}
	/* note. it is assumed that middle = ilo in case ihi = ilo+1 . */
	if (x >= xt[middle])
	    ilo = middle;
	else
	    ihi = middle;
    }
} /* findInterval */

// *********************************************************************
int newFindInterval(double *xt, int n, double x,
		  int rightmost_closed,  int all_inside, int ilo,
		  int *mflag)
// *********************************************************************
{
    int istep, middle, ihi;
#define left_boundary  { *mflag = -1;	return(all_inside ? 1 : 0); }

#define right_boundary { *mflag = +1;					\
	return((all_inside || (rightmost_closed && x == xt[n]))		\
		? (n - 1) : n); }



    if(ilo <= 0) {
	if (x < xt[0]) left_boundary;
	ilo = 1;
    }
    ihi = ilo + 1;
    if (ihi >= n) {
	if (x >= xt[n-1]) right_boundary;
	if (n <= 1) /* x < xt[0] */	left_boundary;
	ilo = n - 1;
	ihi = n;
    }

    if (x < xt[ihi-1]) {
	if (x >= xt[ilo-1]) { /* `lucky': same interval as last time */
	    *mflag = 0;	   return ilo;
	}
	/* **** now x < xt[ilo] .	decrease  ilo  to capture  x */
	for(istep = 1; ; istep *= 2) {
	    ihi = ilo;
	    ilo = ihi - istep;
	    if (ilo <= 1)
		break;
	    if (x >= xt[ilo-1])		goto L50;
	}
	ilo = 1;
	if (x < xt[1-1]) left_boundary;
    }
    else {
	/* **** now x >= xt[ihi-1] .	increase  ihi  to capture  x */
	for(istep = 1; ; istep *= 2) {
	    ilo = ihi;
	    ihi = ilo + istep;
	    if (ihi >= n)
		break;
	    if (x < xt[ihi-1])		goto L50;
	}
	if (x >= xt[n-1]) right_boundary;
	ihi = n;
    }

  L50:
    /* **** now xt[ilo] <= x < xt[ihi] . narrow the interval. */
    for(;;) {
	middle = (ilo + ihi) / 2;
	if (middle == ilo) {
	    *mflag = 0;	   return ilo;
	}
	/* note. it is assumed that middle = ilo in case ihi = ilo+1 . */
	if (x >= xt[middle-1])
	    ilo = middle;
	else
	    ihi = middle;
    }
} // newFindInterval

