#pragma once
#include <dlib/server.h>
#include <fstream>
#include <vector>

using namespace std;
using namespace dlib;

struct HeaderField
{
	std::string key;
	std::string value;
	std::map<std::string, std::string> attributes;
};

class MultipartParser;

class DataParser
{
protected:
    MultipartParser* parser;
public:
	DataParser(MultipartParser* parser) {
        this->parser = parser;
    }

	virtual ~DataParser() {}
    virtual unsigned long read_at_least(unsigned char* buf, unsigned long expect) = 0;
};

class Base64Parser : public DataParser
{
private:
	unsigned int	buffer;
	int				buflen;
	int				pad;

public:
	Base64Parser(MultipartParser* parser)
    : DataParser(parser)
    {
        buflen = 0;
        buffer = 0;
        pad = 0;
    }

    unsigned long read_at_least(unsigned char* buf, unsigned long expect);
};

class QuotedPrintableParser : public DataParser
{
private:
	uint32_t buffer;
	int buflen;
	enum
	{
		NORMAL, SPECIAL
	} state;
public:
	QuotedPrintableParser(MultipartParser* parser) :
	DataParser(parser)
	{
		buflen = 0;
		buffer = 0;
		state = NORMAL;
	}

    unsigned long read_at_least(unsigned char* buf, unsigned long expect);
};

class SimpleParser : public DataParser
{
public:
	SimpleParser(MultipartParser* parser) : DataParser(parser)
    {}

	unsigned long read_at_least(unsigned char* buf, unsigned long expect);
};

class MultipartParser
{
private:
    std::istream& in;
    unsigned long content_length;
    unsigned long consumed;

    std::vector<unsigned char> readbuff;
    size_t buffpos;

	std::string key;
	std::string value;
	std::map<std::string, HeaderField> parsed_fields;

	// States:
	// PRE - Searching for first boundary.
	// ATBOUND - Just matched the whole boundary string.
	// FOLLBOUND - Looking at characters after boundary.
	// STARTOBJECT - Starting a MIME object.
	enum
	{
		PRE, ATBOUND, FOLLBOUND, STARTOBJECT, 
		OBJECT_KEY, OBJECT_PREVAL, OBJECT_CR, OBJECT_EOL, OBJECT_VAL
	} state;

	const std::string boundary;

	void parse_attrs(HeaderField &fld);
	void add_header(const std::string &key, const std::string &value);

    using onFieldCallback = std::function<void(const string& name, const string& filename, const string& value, DataParser* data_parser)>;

    void object_created(const std::map<std::string, HeaderField>& fields, onFieldCallback onField);

public:
	MultipartParser(const string& _boundary, std::istream& _in, unsigned long _content_length);
    void parse(onFieldCallback onField);
    bool get_char(unsigned char& c);
};
