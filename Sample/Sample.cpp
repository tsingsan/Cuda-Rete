#include "ctrans.h"
#include "ReteEngine.h"
#include "RuleReason.h"
#include <locale.h>
#include <iostream>
#include <string>
using namespace std;

const char* RULEPATHPREFIX = "../Rules/";

CReteEngine engine;
int main (int argc, char ** argv)
{
	string rulefile, rdfname = "ABCD";
	if (argc < 2)
	{
		cout<<"Rule Filename: ";
		cin>>rulefile;
		cout<<endl;
	}
	else
	{
		rulefile = argv[1];
	}	
	rulefile = string(RULEPATHPREFIX) + rulefile;

	CRuleReason cr;
	cr.parserule(rulefile.c_str());
	engine.SetRuleReason(&cr);

	int j = 100;
	while (j--)
	{
		cout<<j<<endl;

		//engine.SetResultDifference(false);		
		engine.ParseTriple("http://www.tsingsan.com#A", "http://www.tsingsan.com#Son", "http://www.tsingsan.com#B");	
		engine.ParseTriple("http://www.tsingsan.com#B", "http://www.tsingsan.com#Son", "http://www.tsingsan.com#C");
		engine.ParseTriple("http://www.tsingsan.com#B", "http://www.tsingsan.com#Daughter", "http://www.tsingsan.com#D");
		engine.SetRDFSourceName(rdfname, rulefile);
		engine.Run();
		engine.ClearTemporary();
				
	}
	
	return 0;
}
