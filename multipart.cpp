#include "multipart.h"
#include "utils.h"

const string base64_encoding =	
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "abcdefghijklmnopqrstuvwxyz"
                    "0123456789+/";

unsigned long Base64Parser::read_at_least(unsigned char* buf, unsigned long expect) {
    unsigned long nread = 0;
    
    while (nread  < expect) {
        unsigned char c;
        if (!parser->get_char(c))
            break;

        if (c == '=') {
            buffer = buffer << 6;
            pad += 6;
            buflen++;
        } else {
            size_t pos = base64_encoding.find(c);
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
                buf[nread++] = c;
                buffer = buffer << 8;
            }
            buflen = 0;
            buffer = 0;
            pad = 0;
        }
    }
    return nread;
}

unsigned long QuotedPrintableParser::read_at_least(unsigned char* buf, unsigned long expect) {
    unsigned long nread = 0;
    
    while (nread  < expect) {
        unsigned char c;
        if (!parser->get_char(c))
            break;

        if (state == SPECIAL) {
            static const std::string v = "0123456789ABCDEF";
            if (v.find(c) == std::string::npos) continue;
            buffer = buffer << 8 | (v.find(c));
            buflen++;
            if (buflen == 2) {
                c = (unsigned char) buffer;
                buf[nread++] = c;
                state = NORMAL;
            }
            continue;
        }

        if (c == '=') {
            state = SPECIAL;
            buflen = 0;
            continue;
        }

        // Normal character, output it
        buf[nread++] = c;
    }

    return nread;
}

unsigned long SimpleParser::read_at_least(unsigned char* buf, unsigned long expect) {
    unsigned long nread = 0;
    
    while (nread  < expect) {
        unsigned char c;
        if (!parser->get_char(c))
            break;

        buf[nread++] = c;
    }

    return nread;
}

bool MultipartParser::get_char(unsigned char& c) {
    if (buffpos < readbuff.size()) {
        c = readbuff[buffpos++];
        if (buffpos == readbuff.size()) {
            readbuff.clear();
            buffpos = 0;
        }
        return true;
    }

    // buffer empty
    if (!in.good() || consumed >= content_length) {
        return false;
    }

    unsigned char cdata = in.get();
    consumed++;

    if (cdata != boundary[readbuff.size()]) {
        c = cdata;
        return true;
    }

    readbuff.push_back(cdata);

    while (readbuff.back() == boundary[readbuff.size() - 1]) {
        
        if (readbuff.size() == boundary.size()) {
            readbuff.clear();
            state = ATBOUND;
            return false;
        }
        if (!in.good() || consumed >= content_length) {
            break;
        }
        cdata = in.get();
        consumed++;
        readbuff.push_back(cdata);
    }

    c = readbuff[buffpos++];
    if (buffpos == readbuff.size()) {
        readbuff.clear();
        buffpos = 0;
    }
  
    return true;
}

MultipartParser::MultipartParser(const string& _boundary, std::istream& _in, unsigned long _content_length)
: boundary(_boundary)
, in(_in)
, content_length(_content_length)
, consumed(0)
, buffpos(0)
{
    state = PRE;
}

void MultipartParser::parse(onFieldCallback onField) {
    while (true) {
        unsigned char c;
        if (!get_char(c)) {
            // EOF
            if (state != ATBOUND) {
                return;
            }
            continue;
        }

        switch (state) {
            case ATBOUND: {
                if (c == '-') {
                    state = FOLLBOUND;
                }
                else if (c == '\r') {
                    state = FOLLBOUND;
                }
                else if (c == '\n') {
                    state = STARTOBJECT;
                }
                else throw std::logic_error("MIME duff boundary stuff");
                break;
            }
            case FOLLBOUND: {
                if (c == '-') {
                    // done !
                    return;
                }
                else if (c == '\n') {
                    state = STARTOBJECT;
                }
                else throw std::logic_error("MIME duff boundary stuff");
                break;
            }
            case STARTOBJECT: {
                parsed_fields.clear();

                if (c == ' ' || c == '\t') break;
                if (c == ':') throw std::logic_error("Zero length key?");
                state = OBJECT_KEY;
                key = std::tolower(c);
                break;
            }
            case OBJECT_KEY: {
                if (c == ' ' || c == '\t') break;
                if (c == ':') {
                    state = OBJECT_PREVAL;
                    break;
                }

                key += std::tolower(c);
                break;
            }
            case OBJECT_PREVAL: {
                if (c == ' ' || c == '\t') break;
                if (c == '\r') {
                    state = OBJECT_CR;
                    break;
                }
                if (c == '\n') {
                    state = OBJECT_EOL;
                    break;
                }
                value = c;
                state = OBJECT_VAL;
                break;
            }
            case OBJECT_VAL: {
                if (c == '\r') {
                    state = OBJECT_CR;
                    break;
                }
                if (c == '\n') {
                    state = OBJECT_EOL;
                    break;
                }

                value += c;
                break;
            }
            case OBJECT_CR: {
                if (c == '\n') {
                    state = OBJECT_EOL;
                }
                else throw std::logic_error("CR without LF in MIME header?");
                break;
            }
            case OBJECT_EOL: {
                // This deals with continuation.
                if (c == ' ' || c == '\t') {
                    state = OBJECT_VAL;
                    break;
                }
                // Not a continue, save key/value if there is one.
                if (key != "") {
                    add_header(key, value);
                    key = "";
                    value = "";
                }

                if (c == '\r') {
                    break; 
                }

                if (c == '\n') {
                    object_created(parsed_fields, onField);
                    break;
                }

                // EOL, now reading a key.
                state = OBJECT_KEY;
                key = std::tolower(c);
                break;
            }
        }
    }
}

void MultipartParser::object_created(const std::map<std::string, HeaderField>& fields, onFieldCallback onField) {
    string field_name;
    string file_name;
    string field_value;

    for (const auto& field : fields) {
        if (field.first == "content-disposition" && field.second.value == "form-data") {
            for (const auto& attr : field.second.attributes) {
                if (attr.first == "name") {
                    field_name = attr.second;
                } else if (attr.first == "filename") {
                    file_name = attr.second;
                }
            }
        }
    }

    DataParser *data_parser;

    // Use the parser factory to get an appropriate parser.
    std::string encoding =
                    parsed_fields["content-transfer-encoding"].value;

    if (encoding == "base64") {
        data_parser = new Base64Parser(this);
    }
    else if (encoding == "quoted-printable") {
        data_parser = new QuotedPrintableParser(this);
    }
    else {
        data_parser = new SimpleParser(this);
    }

    if (file_name.size()) {
        onField(field_name, file_name, "", data_parser);
    } else {
        unsigned char buf[128 + 4];
        while (true) {
            auto nread = data_parser->read_at_least(buf, 128);
            field_value += string((char*)buf, nread);
            if (nread < 128) 
                break;
        }
        onField(field_name, "", field_value, nullptr);
    }
    delete data_parser;
}

void MultipartParser::add_header(const std::string &key, const std::string &value) {

    HeaderField fld;

	fld.key = key;
	fld.value = value;

	if (key == "content-type")
		parse_attrs(fld);

    if (key == "content-disposition")
		parse_attrs(fld);

	parsed_fields[key] = fld;
}

static void split(std::string &input, std::string &left, std::string &right, unsigned char spl)
    {
	int pos = input.find(spl);

	if (pos >= 0) {
		right = input.substr(pos + 1);
		left = input.substr(0, pos);
	} else {
		left = input;
		right = "";
	}
    }

void MultipartParser::parse_attrs(HeaderField &fld)
    {
	std::string a, b;

	while (*(fld.key.begin()) == ' ')
		fld.key = fld.key.substr(1);
	while (fld.key[fld.key.size() - 1] == ' ')
		fld.key.erase(fld.key.end() - 1);

	while (*(fld.value.begin()) == ' ')
		fld.value = fld.value.substr(1);
	while (fld.value[fld.value.size() - 1] == ' ')
		fld.value.erase(fld.value.end() - 1);

	split(fld.value, a, b, ';');

	fld.value = a;

	a = b;

	while (a != "") {
		std::string c, d;

		split(a, a, b, ';');

		split(a, c, d, '=');

		while (*(c.begin()) == ' ')
			c = c.substr(1);
		while (c[c.size() - 1] == ' ')
			c.erase(c.end() - 1);

		while (*(d.begin()) == ' ')
			d = d.substr(1);
		while (d[d.size() - 1] == ' ')
			d.erase(d.end() - 1);

		while (*(d.begin()) == '"')
			d = d.substr(1);
		while (d[d.size() - 1] == '"')
			d.erase(d.end() - 1);

		fld.attributes[c] = d;

		a = b;
	}
}

