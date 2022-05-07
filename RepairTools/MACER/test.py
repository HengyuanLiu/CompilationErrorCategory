import base64, clang.cindex, collections, traceback

from clang.cindex import Config
from clang.cindex import Diagnostic

from srcT.DataStruct.Code import Code
from srcT.Common import ConfigFile as CF, Helper as H
with open('data/for_research/test.c',mode='r') as fp:
     codeText = fp.read()

# predCodeObj = Code(codeText)
# predCodeObj.getNumErrors()

# ClangArgs = ['-static', '-Wall', '-funsigned-char', '-Wno-unused-result', '-O', '-Wextra', '-std=c99', "-I/usr/lib/clang/6.0/include", '-I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/']
ClangArgs = ['-static', '-funsigned-char', '-Wno-unused-result', '-O', '-Wextra', '-std=c99', "-I/usr/lib/clang/6.0/include", '-I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/']
index = clang.cindex.Index.create()

filename = 'data/input/temp.c'
index.parse(filename, args=CF.ClangArgs, unsaved_files=[(filename, codeText)])  # 他喵的编译器出问题了？？？
