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

/**
     * Interface describes the events called as a MIME string is parsed.
     * To make the MIME decoder to do something useful, implement all these methods.
     */
class ClientInterface
{
public:
	virtual void object_created(const std::map<std::string, HeaderField>& fields) = 0;
	virtual void data(unsigned char *data, int len) = 0;
	virtual void data_end() = 0;
};


class DataParser
{
public:
	DataParser(ClientInterface *client) {
        this->client = client;
    }

	virtual ~DataParser() {}
	virtual void parse(unsigned char c) = 0;

	ClientInterface *client;
};

class Base64Parser : public DataParser
{
private:
	std::string		encoding;
	unsigned int	buffer;
	int				buflen;
	int				pad;

public:
	Base64Parser(ClientInterface *client)
    : DataParser(client)
    {
        buflen = 0;
        buffer = 0;
        pad = 0;
        encoding =	"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "abcdefghijklmnopqrstuvwxyz"
                    "0123456789+/";
    }

	virtual void parse(unsigned char c)
    {
        if (c == '=') {
            buffer = buffer << 6;
            pad += 6;
            buflen++;
        } else {
            size_t pos = encoding.find(c);
            if (std::string::npos != pos) {
                buffer = buffer << 6;
                buffer = buffer | (pos & 0x3f);
                buflen++;
            }
        }

        if (buflen == 4) {
            for (int i = 0; i < 3; i++) {
                if ((i * 8) > (16 - pad)) continue;
                unsigned char c = (buffer & 0xff0000) >> 16;
                client->data(&c, 1);
                buffer = buffer << 8;
            }
            buflen = 0;
            buffer = 0;
            pad = 0;
        }
    }
};

/**
   \todo The quoted printable parser doesn't work.
*/
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
	QuotedPrintableParser(ClientInterface *client) :
	DataParser(client)
	{
		buflen = 0;
		buffer = 0;
		state = NORMAL;
	}

	virtual void parse(unsigned char c)
    {
        if (state == SPECIAL) {
            static const std::string v = "0123456789ABCDEF";
            if (v.find(c) == std::string::npos) return;
            buffer = buffer << 8 | (v.find(c));
            buflen++;
            if (buflen == 2) {
                c = (unsigned char) buffer;
                client->data(&c, 1);
                state = NORMAL;
            }
            return;
        }

        if (c == '=') {
            state = SPECIAL;
            buflen = 0;
            return;
        }

        // Normal character, output it
        client->data(&c, 1);
    }
};

class SimpleParser : public DataParser
{
public:
	SimpleParser(ClientInterface *client) : DataParser(client)
  {}

	virtual void parse(unsigned char c)
    {
        client->data(&c, 1);
    }
};

class MultipartParser
{
private:
	ClientInterface *client;
	DataParser *data_parser;

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
		PRE, ATBOUND, FOLLBOUND, DONE, STARTOBJECT, 
		OBJECT_PREKEY, OBJECT_KEY, OBJECT_PREVAL, OBJECT_CR, OBJECT_EOL, OBJECT_VAL,
		OBJECT_HEADER_COMPLETE, OBJECT_BODY
	} state;

	const std::string boundary;
	size_t posn;

	std::vector<unsigned char> buffered;

	void parse_attrs(HeaderField &fld);
	void add_header(const std::string &key, const std::string &value);
	void parseSection(unsigned char c);

public:
	MultipartParser(ClientInterface *client, const string& _boundary);
	void parse(unsigned char c);
	void close();
};

class MultipartHandler : public ClientInterface
{
public:
  struct file_node {
    string field_name;
    string name;
    string path;
  };

  std::vector<file_node> files;

private:
  incoming_things& incoming;
  string dest_dir;

    struct {
      string field_name;
      string file_name;
      string buffer;
      string path;
      ofstream ofs;
  } part;

public:
	MultipartHandler(incoming_things& _incoming, const string& _dest_dir): incoming(_incoming), dest_dir(_dest_dir) {}

	virtual void object_created(const std::map<std::string, HeaderField>& fields);
	virtual void data(unsigned char *data, int len);
	virtual void data_end();
};
