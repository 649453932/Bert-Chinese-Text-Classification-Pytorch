#pragma once
// 本文件源码改编自 https://gist.github.com/luistung/ace4888cf5fd1bad07844021cb2c7ecf
// 感谢
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

//https://unicode.org/reports/tr15/#Norm_Forms
//https://ssl.icu-project.org/apiref/icu4c/uchar_8h.html

const std::wstring stripChar = L" \t\n\r\v\f";
using Vocab = std::unordered_map<std::wstring, size_t>;
using InvVocab = std::unordered_map<size_t, std::wstring>;

class BasicTokenizer {
public:
    BasicTokenizer(bool doLowerCase);
    std::vector<std::wstring> tokenize(const std::string& text) const;

private:
    std::wstring cleanText(const std::wstring& text) const;
    bool isControol(const wchar_t& ch) const;
    bool isWhitespace(const wchar_t& ch) const;
    bool isPunctuation(const wchar_t& ch) const;
    bool isChineseChar(const wchar_t& ch) const;
    std::wstring tokenizeChineseChars(const std::wstring& text) const;
    bool isStripChar(const wchar_t& ch) const;
    std::wstring strip(const std::wstring& text) const;
    std::vector<std::wstring> split(const std::wstring& text) const;
    std::wstring runStripAccents(const std::wstring& text) const;
    std::vector<std::wstring> runSplitOnPunc(const std::wstring& text) const;

    bool mDoLowerCase;
};

class WordpieceTokenizer {
public:
    WordpieceTokenizer(std::shared_ptr<Vocab> vocab, const std::wstring& unkToken = L"[UNK]", size_t maxInputCharsPerWord=200);
    std::vector<std::wstring> tokenize(const std::wstring& text) const;

private:
    std::shared_ptr<Vocab> mVocab;
    std::wstring mUnkToken;
    size_t mMaxInputCharsPerWord;
};

class FullTokenizer {
public:
    FullTokenizer(const std::string& vocabFile, bool doLowerCase = true);
    std::vector<std::wstring> tokenize(const std::string& text) const;
    std::vector<size_t> convertTokensToIds(const std::vector<std::wstring>& text) const;

private:
    std::shared_ptr<Vocab> mVocab;
    InvVocab mInvVocab;
    std::string mVocabFile;
    bool mDoLowerCase;
    BasicTokenizer mBasicTokenizer;
    WordpieceTokenizer mWordpieceTokenizer;
};
