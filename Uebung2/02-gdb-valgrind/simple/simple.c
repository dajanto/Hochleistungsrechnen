/*
** simple error demonstration to demonstrate power of valgrind
** Julian M. Kunkel - 17.04.2008
*/

#include <stdio.h>
#include <stdlib.h>

int *
mistake1 ()
{
  int *buf = malloc (sizeof (int) * 6);
  buf[0] = 1;
  buf[1] = 1;
  buf[2] = 2;
  buf[3] = 3;
  buf[4] = 4;
  buf[5] = 5;
  //int buf[] = { 1, 1, 2, 3, 4, 5 };     //buf[6] ohne die 6 funktioniert das aus irgendeinem Grund nicht richtig?
  return buf;                            //https://stackoverflow.com/questions/50083744/how-to-create-an-array-without-declaring-the-size-in-c/50084026#50084026
}

int *
mistake2 ()
{
  int *buf = malloc (sizeof (int) * 4);
  buf[1] = 2;
  return buf;
}


int *
mistake3 ()
{
  /* In dieser Funktion darf kein Speicher direkt allokiert werden. */
  //int mistake2_ = 0;
  int *buf = (int *) mistake2();
  buf[0] = 3;
  return buf;
}


int *
mistake4 ()
{
  int *buf = malloc (sizeof (int) * 4);
  buf[0] = 4;
  //free (buf);
  return buf;
}



int
main (void)
{
  /* Modifizieren Sie die folgende Zeile nicht */
  int *p[4] = { &mistake1 ()[1], &mistake2 ()[1], mistake3 (), mistake4 () };

  printf ("1 %d\n", *p[0]);
  printf ("2 %d\n", *p[1]);
  printf ("3 %d\n", *p[2]);
  printf ("4 %d\n", *p[3]);

  /* mhh muss hier noch etwas gefreed werden? */
  /* Fügen sie hier die korrekten aufrufe von free() ein */
//  free (p[1]);			/* welcher Pointer war das doch gleich?, TODO: Fixme... :-) */
  free (p[0] -1);
  free (p[1] -1);
  free (p[2]);
  free (p[3]);
  return 0;
}
