
#include <stdio.h> 
#include <stdlib.h> 
float fact ( float n ){ 
float i ; 
float ans = n ; 
int final = 0 ; 
for ( i = 0 ; i <= n ; i ++){ 
ans = ans / i ; 
if ( ans > 0 ){ 
continue ;} 
if ( ans = 0 ){ 
final = 0 ; 
break ;} 
if ( ans < 0 ){ 
final = 0 ; 
break ;}} 
return ans ;} 
int main (){ 
float n = 0 ; 
int d = 0 / 0 ; 
printf ( "String" , d ); 
printf ( "String" , fact ( 0 )); 
return 0 ;} 