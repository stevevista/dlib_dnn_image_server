#include "multipart.h"
#include "utils.h"

MultipartParser::MultipartParser(ClientInterface *client, const string& _boundary)
: boundary(_boundary)
{
	this->client = client;

    state = PRE;
    posn = 0;
    data_parser = 0;
}

void MultipartParser::parse(unsigned char c) {

    if (state == PRE) {
        if (c == boundary[posn])
            posn++;
        else {
            if (c == boundary[0])
                posn = 1;
            else
                posn = 0;
        }

        if (posn == boundary.size()) {
            state = ATBOUND;
            posn = 0;
        }

        return;
    }
        
    if (state == ATBOUND) {
        if (c == '-') {
            state = FOLLBOUND;
            return;
        }
        if (c == '\r') {
            state = FOLLBOUND;
            return;
        }
        if (c == '\n') {
            state = STARTOBJECT;
            return;
        }
        throw std::logic_error("MIME duff boundary stuff");
    }

    if (state == FOLLBOUND) {
        if (c == '-') {
            state = DONE;
            return;
        }
        if (c == '\n') {
            state = STARTOBJECT;
            return;
        }
        throw std::logic_error("MIME duff boundary stuff");
    }

    if (state == STARTOBJECT) {

        if (data_parser) {
			client->data_end();
            delete data_parser;
            data_parser = 0;
        }

        parsed_fields.clear();
        state = OBJECT_PREKEY;
        posn = 0;
        buffered.clear();
        // go through
    }

    if (state == OBJECT_PREKEY) {
        if (c == ' ' || c == '\t') return;
		if (c == ':') throw std::logic_error("Zero length key?");
		state = OBJECT_KEY;
		key = tolower(c);
        return;
    }

    if (state > OBJECT_PREKEY) {
        if (c == boundary[posn]) {
            posn++;
            if (posn == boundary.size()) {
                state = ATBOUND;
                posn = 0;
                return;
            }
            buffered.push_back(c);
        } else {
            // Dispose of buffer
            if (posn) {
                for (auto bc : buffered)
                    parseSection(bc);
                buffered.clear();
                posn = 0;
            }

            // This character may start the match again.
            if (c == boundary[0]) {
                posn = 1;
                buffered.push_back(c);
            } else {
                posn = 0;
                parseSection(c);
            }
        }
        return;
    }


}

void MultipartParser::parseSection(unsigned char c) {
    if (state == OBJECT_KEY) {
        if (c == ' ' || c == '\t') return;
        if (c == ':') {
            state = OBJECT_PREVAL;
            return;
        }

        key += tolower(c);
        return;
    }

    if (state == OBJECT_PREVAL) {
                    if (c == ' ' || c == '\t') return;
                    if (c == '\r') {
                        state = OBJECT_CR;
                        return;
                    }
                    if (c == '\n') {
                        state = OBJECT_EOL;
                        return;
                    }
			        value = c;
			        state = OBJECT_VAL;
			        return;
                }
                
    if (state == OBJECT_VAL) {
        if (c == '\r') {
            state = OBJECT_CR;
            return;
        }
        if (c == '\n') {
            state = OBJECT_EOL;
            return;
        }

        value += c;
        return;
    }
                
    if (state == OBJECT_CR) {
        if (c == '\n') {
            state = OBJECT_EOL;
            return;
        }
        throw std::logic_error("CR without LF in MIME header?");
    }
                
    if (state == OBJECT_EOL) {
        // This deals with continuation.
        if (c == ' ' || c == '\t') {
            state = OBJECT_VAL;
            return;
        }

        // Not a continue, save key/value if there is one.
        if (key != "") {
            add_header(key, value);
            key = "";
            value = "";
        }

        if (c == '\r') {
            return; 
        }

        if (c == '\n') {
            state = OBJECT_HEADER_COMPLETE;
            client->object_created(parsed_fields);
            return;
        }

        // EOL, now reading a key.
        state = OBJECT_KEY;
        key = tolower(c);
        return;
    }

    if (state == OBJECT_HEADER_COMPLETE) {
        if (data_parser)
            delete data_parser;

        // Use the parser factory to get an appropriate parser.
        std::string encoding =
            parsed_fields["content-transfer-encoding"].value;

        if (encoding == "base64") {
            data_parser = new Base64Parser(client);
        }
        else if (encoding == "quoted-printable") {
            data_parser = new QuotedPrintableParser(client);
        }
        else {
            data_parser = new SimpleParser(client);
        }

        state = OBJECT_BODY;
        // go through
    }
                
    if (state == OBJECT_BODY) {
        data_parser->parse(c);
        return;
    }
}

void MultipartParser::close()
{
    // Dispose of buffer
    if (posn) {
        for (auto bc : buffered)
            parseSection(bc);
        buffered.clear();
        posn = 0;
    }

    // Dispose of buffer
    if (data_parser) {
        client->data_end();
        delete data_parser;
    }
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

void MultipartHandler::object_created(const std::map<std::string, HeaderField>& fields) {
  part.field_name = "";
  part.file_name = "";
  part.buffer = "";

  for (const auto& field : fields) {
        if (field.first == "content-disposition" && field.second.value == "form-data") {
            for (const auto& attr : field.second.attributes) {
                if (attr.first == "name") {
                    part.field_name = attr.second;
                } else if (attr.first == "filename") {
                    part.file_name = attr.second;
                }
            }
        }
  }

  if (part.file_name.size()) {
    auto parts = split_path(part.file_name);
    auto storename = random_string() + parts[2];
    part.path = dest_dir + storename;
    part.ofs.open(part.path, ofstream::binary);
  }

}

void MultipartHandler::data_end()
{
    if (part.file_name.empty()) {
        incoming.queries[part.field_name] = part.buffer;
    } else {
      part.ofs.close();
      files.push_back({part.field_name, part.file_name, part.path});
    }
}

void MultipartHandler::data(unsigned char *data, int len)
{
    if (part.file_name.empty()) {
        part.buffer += string((char*)data, len);
    } else {
      part.ofs.write((const char*)data, len);
    }
}
