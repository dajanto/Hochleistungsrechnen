/* 
** simple error demonstration to demonstrate power of valgrind
** Julian M. Kunkel - 17.04.2008
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int *
mistake1 ()
{
  /* 
    buf muss dynamisch allokiert werden, um es als Pointer von der Funktion
    zurückzugeben. Es kommt sonst zu einem invaliden Lesen von Speicher,
    da der Stack, auf dem buf allokiert wurde, "gelöscht" ist.
   */
  int *buf = malloc(sizeof(int) * 2);
  buf[1] = 1;
  return buf;
}

int *
mistake2 ()
{
  /* 
    Für die Allokation von Speicher sollte auch der dazugehörige Pointer-Typ
    benutzt werden, da sonst eventuell nicht genügen Speicher allokiert wird.
    Also sizeof (int) statt sizeof (char).
  */
  int *buf = malloc (sizeof (int) * 2);
  buf[1] = 2;
  return buf;
}

int *
mistake3 ()
{
  /* In dieser Funktion darf kein Speicher direkt allokiert werden. */
  /* 
    int mistake2_ = 0;
    int *buf = (int *) &mistake2;
    mistake2 ist ein Funktion Pointer und darf daher nicht beschrieben werden.
   */
  int *buf = mistake2();
  buf[0] = 3;
  return buf;
}

int *
mistake4 ()
{
  /* 
    Für die Allokation von Speicher sollte auch der dazugehörige Pointer-Typ
    benutzt werden, da sonst eventuell nicht genügen Speicher allokiert wird.
    Also sizeof (int) statt sizeof (char).
  */
  int *buf = malloc (sizeof (int) * 4);
  /* Damit der Wert ausgegeben wird, muss er auch auf dem richtigen Index liegen */
  buf[0] = 4;
  /* buf darf nicht freigegeben werden, wenn es von der Funktion zurückgegeben werden soll */
  /* free (buf); */
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

  /* 
    Pointer muss dekrementiert werden, da er zum ersten, nicht nullten Element zeigt.
    Der Pointer muss zum nullten Element zeigen, damit der Speicher des Array vollständig
    befreit werden kann.
  */
  free (--p[0]); 
  /* 
    Pointer muss dekrementiert werden, da er zum ersten, nicht nullten Element zeigt.
    Der Pointer muss zum nullten Element zeigen, damit der Speicher des Array vollständig
    befreit werden kann.
  */
  free (--p[1]);
  free (p[2]); /* Pointer wurde dynamisch allokiert, muss auch befreit werden */
  free (p[3]); /* Pointer wurde dynamisch allokiert, muss auch befreit werden */
  return 0;
}
