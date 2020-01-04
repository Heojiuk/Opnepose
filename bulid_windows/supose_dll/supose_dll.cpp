// supose_dll.cpp : DLL 응용 프로그램을 위해 내보낸 함수를 정의합니다.
//


#include "stdafx.h"

__declspec(dllexport) int Sum(int a, int b) {
	return a + b;

	
}

int main() {
	
	return 0;
}