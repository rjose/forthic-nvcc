#pragma once
#include <string>

class TokenizerTest {
public:
    TokenizerTest();
    void run();

private:
    void testWhitespace();
    void testComment();
    void printFailure(bool pass, const char* testName);
    void testStartEndDefinition();
    void testStartEndArray();
    void testStartEndNamedModule();
    void testAnonymousModule();
    void testTripleQuote();
    void testTripleQuoteString();
    void testString();
    void testWord();

private:
   std::string name; 
};
