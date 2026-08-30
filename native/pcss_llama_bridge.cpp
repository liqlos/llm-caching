// Minimal, process-isolated llama.cpp C-API bridge for verifier correctness.
//
// This executable intentionally reports no timing, throughput, memory, or
// hardware suitability metric.  It accepts raw token IDs only and uses a
// deterministic greedy argmax solely to make serial and batched state
// reconstruction comparable.

#include <llama.h>

#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#endif

#include <algorithm>
#include <chrono>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char * kRequestSchema = "pcss.llama-bridge-request/v1";
constexpr const char * kResultSchema = "pcss.llama-bridge-result/v1";
constexpr const char * kCapabilitiesSchema = "pcss.llama-bridge-capabilities/v2";
constexpr const char * kReferenceSchema = "pcss.llama-bridge-reference/v2";
constexpr const char * kPendingReferenceSchema = "pcss.llama-bridge-pending-reference/v1";
constexpr const char * kVerifySchema = "pcss.llama-bridge-verify/v2";
constexpr const char * kPendingVerifySchema = "pcss.llama-bridge-pending-verify/v1";
constexpr const char * kPendingTrajectorySchema = "pcss.llama-bridge-pending-trajectory/v1";
constexpr const char * kProfileSchema = "pcss.llama-bridge-profile/v1";
constexpr const char * kPendingProfileSchema = "pcss.llama-bridge-pending-profile/v1";
constexpr const char * kPendingBenchmarkSchema = "pcss.llama-bridge-pending-benchmark/v1";
constexpr const char * kStateExport = "model-architecture+memory-state/v2";
constexpr const char * kStateApi = "llama_state_get_data+llama_state_set_data/model-architecture+memory/v2";
constexpr std::size_t kMaxRequestLineBytes = 16U * 1024U * 1024U;

class BridgeError final : public std::runtime_error {
  public:
    BridgeError(std::string code, std::string message)
        : std::runtime_error(std::move(message)), code_(std::move(code)) {}

    const std::string & code() const noexcept {
        return code_;
    }

  private:
    std::string code_;
};

struct Json {
    enum class Kind {
        null_value,
        boolean,
        integer,
        string,
        array,
        object,
    };

    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;

    Kind kind = Kind::null_value;
    bool boolean = false;
    std::int64_t integer = 0;
    std::string text;
    Array array;
    Object object;

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool value) : kind(Kind::boolean), boolean(value) {}
    Json(std::int64_t value) : kind(Kind::integer), integer(value) {}
    Json(const char * value) : kind(Kind::string), text(value) {}
    Json(std::string value) : kind(Kind::string), text(std::move(value)) {}
    Json(Array value) : kind(Kind::array), array(std::move(value)) {}
    Json(Object value) : kind(Kind::object), object(std::move(value)) {}
};

class JsonParser final {
  public:
    explicit JsonParser(const std::string & input) : input_(input) {}

    Json parse_document() {
        skip_whitespace();
        Json value = parse_value();
        skip_whitespace();
        if (position_ != input_.size()) {
            fail("unexpected trailing JSON data");
        }
        return value;
    }

  private:
    const std::string & input_;
    std::size_t position_ = 0;

    [[noreturn]] static void fail(const char * message) {
        throw BridgeError("invalid_json", message);
    }

    void skip_whitespace() {
        while (position_ < input_.size()) {
            const unsigned char value = static_cast<unsigned char>(input_[position_]);
            if (value != ' ' && value != '\n' && value != '\r' && value != '\t') {
                break;
            }
            ++position_;
        }
    }

    char peek() const {
        if (position_ >= input_.size()) {
            fail("unexpected end of JSON input");
        }
        return input_[position_];
    }

    bool consume(char expected) {
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected)) {
            fail("malformed JSON");
        }
    }

    Json parse_value() {
        skip_whitespace();
        switch (peek()) {
            case '{':
                return parse_object();
            case '[':
                return parse_array();
            case '"':
                return Json(parse_string());
            case 't':
                consume_literal("true");
                return Json(true);
            case 'f':
                consume_literal("false");
                return Json(false);
            case 'n':
                consume_literal("null");
                return Json(nullptr);
            default:
                if (peek() == '-' || std::isdigit(static_cast<unsigned char>(peek())) != 0) {
                    return Json(parse_integer());
                }
                fail("unsupported JSON value");
        }
    }

    void consume_literal(const char * literal) {
        for (const char * current = literal; *current != '\0'; ++current) {
            if (position_ >= input_.size() || input_[position_] != *current) {
                fail("malformed JSON literal");
            }
            ++position_;
        }
    }

    Json parse_object() {
        expect('{');
        skip_whitespace();
        Json::Object result;
        if (consume('}')) {
            return Json(std::move(result));
        }
        while (true) {
            skip_whitespace();
            if (peek() != '"') {
                fail("JSON object key must be a string");
            }
            std::string key = parse_string();
            skip_whitespace();
            expect(':');
            Json value = parse_value();
            if (!result.emplace(std::move(key), std::move(value)).second) {
                fail("duplicate JSON object key");
            }
            skip_whitespace();
            if (consume('}')) {
                break;
            }
            expect(',');
        }
        return Json(std::move(result));
    }

    Json parse_array() {
        expect('[');
        skip_whitespace();
        Json::Array result;
        if (consume(']')) {
            return Json(std::move(result));
        }
        while (true) {
            result.push_back(parse_value());
            skip_whitespace();
            if (consume(']')) {
                break;
            }
            expect(',');
        }
        return Json(std::move(result));
    }

    static std::uint32_t hex_digit(char value) {
        if (value >= '0' && value <= '9') {
            return static_cast<std::uint32_t>(value - '0');
        }
        if (value >= 'a' && value <= 'f') {
            return static_cast<std::uint32_t>(value - 'a' + 10);
        }
        if (value >= 'A' && value <= 'F') {
            return static_cast<std::uint32_t>(value - 'A' + 10);
        }
        fail("invalid JSON unicode escape");
    }

    std::uint32_t parse_unicode_escape() {
        if (position_ + 4U > input_.size()) {
            fail("truncated JSON unicode escape");
        }
        std::uint32_t value = 0;
        for (int index = 0; index < 4; ++index) {
            value = (value << 4U) | hex_digit(input_[position_++]);
        }
        return value;
    }

    static void append_utf8(std::string & output, std::uint32_t codepoint) {
        if (codepoint <= 0x7fU) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ffU) {
            output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0xffffU) {
            output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0x10ffffU) {
            output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else {
            fail("invalid JSON unicode codepoint");
        }
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (true) {
            if (position_ >= input_.size()) {
                fail("unterminated JSON string");
            }
            const unsigned char value = static_cast<unsigned char>(input_[position_++]);
            if (value == '"') {
                return result;
            }
            if (value < 0x20U) {
                fail("control character in JSON string");
            }
            if (value != '\\') {
                result.push_back(static_cast<char>(value));
                continue;
            }
            if (position_ >= input_.size()) {
                fail("truncated JSON escape");
            }
            const char escaped = input_[position_++];
            switch (escaped) {
                case '"':
                case '\\':
                case '/':
                    result.push_back(escaped);
                    break;
                case 'b':
                    result.push_back('\b');
                    break;
                case 'f':
                    result.push_back('\f');
                    break;
                case 'n':
                    result.push_back('\n');
                    break;
                case 'r':
                    result.push_back('\r');
                    break;
                case 't':
                    result.push_back('\t');
                    break;
                case 'u': {
                    std::uint32_t codepoint = parse_unicode_escape();
                    if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
                        if (position_ + 2U > input_.size() || input_[position_] != '\\' ||
                            input_[position_ + 1U] != 'u') {
                            fail("unpaired high surrogate in JSON string");
                        }
                        position_ += 2U;
                        const std::uint32_t low = parse_unicode_escape();
                        if (low < 0xdc00U || low > 0xdfffU) {
                            fail("invalid low surrogate in JSON string");
                        }
                        codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) + (low - 0xdc00U);
                    } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
                        fail("unpaired low surrogate in JSON string");
                    }
                    append_utf8(result, codepoint);
                    break;
                }
                default:
                    fail("invalid JSON escape");
            }
        }
    }

    std::int64_t parse_integer() {
        const std::size_t begin = position_;
        consume('-');
        if (position_ >= input_.size()) {
            fail("truncated JSON number");
        }
        if (input_[position_] == '0') {
            ++position_;
            if (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
                fail("leading zero in JSON number");
            }
        } else {
            if (std::isdigit(static_cast<unsigned char>(input_[position_])) == 0) {
                fail("invalid JSON number");
            }
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
                ++position_;
            }
        }
        if (position_ < input_.size() && (input_[position_] == '.' || input_[position_] == 'e' ||
                                          input_[position_] == 'E')) {
            fail("non-integer JSON numbers are unsupported");
        }
        std::int64_t value = 0;
        const auto conversion = std::from_chars(
            input_.data() + static_cast<std::ptrdiff_t>(begin),
            input_.data() + static_cast<std::ptrdiff_t>(position_),
            value
        );
        if (conversion.ec != std::errc{} || conversion.ptr != input_.data() + static_cast<std::ptrdiff_t>(position_)) {
            fail("JSON integer is out of range");
        }
        return value;
    }
};

std::string json_escape(const std::string & value) {
    std::string result;
    result.reserve(value.size() + 2U);
    result.push_back('"');
    static constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char character : value) {
        switch (character) {
            case '"':
                result += "\\\"";
                break;
            case '\\':
                result += "\\\\";
                break;
            case '\b':
                result += "\\b";
                break;
            case '\f':
                result += "\\f";
                break;
            case '\n':
                result += "\\n";
                break;
            case '\r':
                result += "\\r";
                break;
            case '\t':
                result += "\\t";
                break;
            default:
                if (character < 0x20U) {
                    result += "\\u00";
                    result.push_back(hex[(character >> 4U) & 0x0fU]);
                    result.push_back(hex[character & 0x0fU]);
                } else {
                    result.push_back(static_cast<char>(character));
                }
                break;
        }
    }
    result.push_back('"');
    return result;
}

std::string json_dump(const Json & value) {
    switch (value.kind) {
        case Json::Kind::null_value:
            return "null";
        case Json::Kind::boolean:
            return value.boolean ? "true" : "false";
        case Json::Kind::integer:
            return std::to_string(value.integer);
        case Json::Kind::string:
            return json_escape(value.text);
        case Json::Kind::array: {
            std::string result = "[";
            for (std::size_t index = 0; index < value.array.size(); ++index) {
                if (index != 0U) {
                    result.push_back(',');
                }
                result += json_dump(value.array[index]);
            }
            result.push_back(']');
            return result;
        }
        case Json::Kind::object: {
            std::string result = "{";
            bool first = true;
            for (const auto & [key, entry] : value.object) {
                if (!first) {
                    result.push_back(',');
                }
                first = false;
                result += json_escape(key);
                result.push_back(':');
                result += json_dump(entry);
            }
            result.push_back('}');
            return result;
        }
    }
    throw BridgeError("internal_error", "unknown JSON value kind");
}

const Json::Object & as_object(const Json & value, const char * field) {
    if (value.kind != Json::Kind::object) {
        throw BridgeError("invalid_request", std::string(field) + " must be a JSON object");
    }
    return value.object;
}

const Json::Array & as_array(const Json & value, const char * field) {
    if (value.kind != Json::Kind::array) {
        throw BridgeError("invalid_request", std::string(field) + " must be a JSON array");
    }
    return value.array;
}

const Json & required_field(const Json::Object & object, const char * field) {
    const auto found = object.find(field);
    if (found == object.end()) {
        throw BridgeError("invalid_request", std::string("missing required field: ") + field);
    }
    return found->second;
}

std::string required_string(const Json::Object & object, const char * field) {
    const Json & value = required_field(object, field);
    if (value.kind != Json::Kind::string || value.text.empty()) {
        throw BridgeError("invalid_request", std::string(field) + " must be a non-empty string");
    }
    return value.text;
}

bool required_boolean(const Json::Object & object, const char * field) {
    const Json & value = required_field(object, field);
    if (value.kind != Json::Kind::boolean) {
        throw BridgeError("invalid_request", std::string(field) + " must be a boolean");
    }
    return value.boolean;
}

std::int64_t required_integer(const Json::Object & object, const char * field) {
    const Json & value = required_field(object, field);
    if (value.kind != Json::Kind::integer) {
        throw BridgeError("invalid_request", std::string(field) + " must be an integer");
    }
    return value.integer;
}

void require_exact_keys(
    const Json::Object & object,
    std::initializer_list<const char *> expected
) {
    if (object.size() != expected.size()) {
        throw BridgeError("invalid_request", "request contains unknown or missing fields");
    }
    for (const char * name : expected) {
        if (object.find(name) == object.end()) {
            throw BridgeError("invalid_request", "request contains unknown or missing fields");
        }
    }
}

void require_request_schema(const Json::Object & object, const char * operation) {
    if (required_string(object, "schema_version") != kRequestSchema) {
        throw BridgeError("invalid_request", "unsupported bridge request schema");
    }
    if (required_string(object, "operation") != operation) {
        throw BridgeError("invalid_request", "request operation does not match its payload");
    }
}

std::uint32_t required_u32(
    const Json::Object & object,
    const char * field,
    std::uint32_t minimum
) {
    const std::int64_t value = required_integer(object, field);
    if (value < static_cast<std::int64_t>(minimum) ||
        value > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw BridgeError("invalid_request", std::string(field) + " is outside the supported range");
    }
    return static_cast<std::uint32_t>(value);
}

std::vector<llama_token> required_tokens(
    const Json::Object & object,
    const char * field
) {
    const Json::Array & values = as_array(required_field(object, field), field);
    if (values.empty()) {
        throw BridgeError("invalid_request", std::string(field) + " must not be empty");
    }
    std::vector<llama_token> result;
    result.reserve(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (values[index].kind != Json::Kind::integer || values[index].integer < 0 ||
            values[index].integer > static_cast<std::int64_t>(std::numeric_limits<llama_token>::max())) {
            throw BridgeError(
                "invalid_request",
                std::string(field) + "[" + std::to_string(index) + "] must be a non-negative llama token ID"
            );
        }
        result.push_back(static_cast<llama_token>(values[index].integer));
    }
    return result;
}

Json token_array(const std::vector<llama_token> & tokens) {
    Json::Array values;
    values.reserve(tokens.size());
    for (const llama_token token : tokens) {
        values.emplace_back(static_cast<std::int64_t>(token));
    }
    return Json(std::move(values));
}

class Sha256 final {
  public:
    Sha256() {
        state_ = {
            0x6a09e667U,
            0xbb67ae85U,
            0x3c6ef372U,
            0xa54ff53aU,
            0x510e527fU,
            0x9b05688cU,
            0x1f83d9abU,
            0x5be0cd19U,
        };
    }

    void update(const std::uint8_t * data, std::size_t length) {
        bit_count_ += static_cast<std::uint64_t>(length) * 8U;
        while (length > 0U) {
            const std::size_t copied = std::min(length, buffer_.size() - buffer_size_);
            std::copy(data, data + copied, buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_));
            buffer_size_ += copied;
            data += copied;
            length -= copied;
            if (buffer_size_ == buffer_.size()) {
                transform(buffer_.data());
                buffer_size_ = 0U;
            }
        }
    }

    std::array<std::uint8_t, 32> finish() {
        const std::uint64_t original_bit_count = bit_count_;
        const std::uint8_t one = 0x80U;
        update(&one, 1U);
        const std::uint8_t zero = 0U;
        while (buffer_size_ != 56U) {
            update(&zero, 1U);
        }
        std::array<std::uint8_t, 8> length{};
        for (std::size_t index = 0; index < length.size(); ++index) {
            length[length.size() - 1U - index] =
                static_cast<std::uint8_t>((original_bit_count >> (index * 8U)) & 0xffU);
        }
        update(length.data(), length.size());

        std::array<std::uint8_t, 32> digest{};
        for (std::size_t index = 0; index < state_.size(); ++index) {
            digest[index * 4U] = static_cast<std::uint8_t>((state_[index] >> 24U) & 0xffU);
            digest[index * 4U + 1U] = static_cast<std::uint8_t>((state_[index] >> 16U) & 0xffU);
            digest[index * 4U + 2U] = static_cast<std::uint8_t>((state_[index] >> 8U) & 0xffU);
            digest[index * 4U + 3U] = static_cast<std::uint8_t>(state_[index] & 0xffU);
        }
        return digest;
    }

  private:
    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_size_ = 0U;
    std::uint64_t bit_count_ = 0U;

    static std::uint32_t rotr(std::uint32_t value, std::uint32_t amount) {
        return (value >> amount) | (value << (32U - amount));
    }

    void transform(const std::uint8_t * block) {
        static constexpr std::array<std::uint32_t, 64> constants = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
            0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
            0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
            0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
            0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
            0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
            0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
            0xc67178f2U,
        };
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16U; ++index) {
            words[index] = (static_cast<std::uint32_t>(block[index * 4U]) << 24U) |
                           (static_cast<std::uint32_t>(block[index * 4U + 1U]) << 16U) |
                           (static_cast<std::uint32_t>(block[index * 4U + 2U]) << 8U) |
                           static_cast<std::uint32_t>(block[index * 4U + 3U]);
        }
        for (std::size_t index = 16U; index < words.size(); ++index) {
            const std::uint32_t small0 = rotr(words[index - 15U], 7U) ^ rotr(words[index - 15U], 18U) ^
                                         (words[index - 15U] >> 3U);
            const std::uint32_t small1 = rotr(words[index - 2U], 17U) ^ rotr(words[index - 2U], 19U) ^
                                         (words[index - 2U] >> 10U);
            words[index] = words[index - 16U] + small0 + words[index - 7U] + small1;
        }
        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const std::uint32_t sum1 = rotr(e, 6U) ^ rotr(e, 11U) ^ rotr(e, 25U);
            const std::uint32_t choice = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + sum1 + choice + constants[index] + words[index];
            const std::uint32_t sum0 = rotr(a, 2U) ^ rotr(a, 13U) ^ rotr(a, 22U);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }
};

std::string sha256_hex(const std::vector<std::uint8_t> & bytes) {
    Sha256 sha;
    sha.update(bytes.data(), bytes.size());
    const std::array<std::uint8_t, 32> digest = sha.finish();
    static constexpr char hex[] = "0123456789abcdef";
    std::string output;
    output.reserve(digest.size() * 2U);
    for (const std::uint8_t byte : digest) {
        output.push_back(hex[(byte >> 4U) & 0x0fU]);
        output.push_back(hex[byte & 0x0fU]);
    }
    return output;
}

std::int32_t required_i32(
    const Json::Object & object,
    const char * field,
    std::int32_t minimum
) {
    const std::int64_t value = required_integer(object, field);
    if (value < static_cast<std::int64_t>(minimum) ||
        value > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw BridgeError("invalid_request", std::string(field) + " is outside the supported range");
    }
    return static_cast<std::int32_t>(value);
}

enum class FlashAttnMode {
    automatic,
    enabled,
    disabled,
};

struct RuntimeConfig {
    std::uint32_t n_ctx = 0U;
    std::uint32_t n_batch = 0U;
    std::int32_t n_gpu_layers = 0;
    FlashAttnMode flash_attn = FlashAttnMode::automatic;
    std::uint32_t seed = 0U;
    std::int32_t n_threads = 0;
    std::int32_t n_threads_batch = 0;
    std::string chat_template;
    std::string thinking_mode;
};

struct RequestConfig {
    std::string model_artifact;
    std::string declared_llama_cpp_library;
    std::string declared_llama_cpp_revision;
    RuntimeConfig runtime;
};

FlashAttnMode parse_flash_attn(const Json::Object & object) {
    const std::string value = required_string(object, "flash_attn");
    if (value == "auto") {
        return FlashAttnMode::automatic;
    }
    if (value == "enabled") {
        return FlashAttnMode::enabled;
    }
    if (value == "disabled") {
        return FlashAttnMode::disabled;
    }
    throw BridgeError("invalid_request", "runtime.flash_attn must be auto, enabled, or disabled");
}

const char * flash_attn_name(FlashAttnMode value) {
    switch (value) {
        case FlashAttnMode::automatic:
            return "auto";
        case FlashAttnMode::enabled:
            return "enabled";
        case FlashAttnMode::disabled:
            return "disabled";
    }
    throw BridgeError("internal_error", "unknown flash attention mode");
}

enum llama_flash_attn_type flash_attn_type(FlashAttnMode value) {
    switch (value) {
        case FlashAttnMode::automatic:
            return LLAMA_FLASH_ATTN_TYPE_AUTO;
        case FlashAttnMode::enabled:
            return LLAMA_FLASH_ATTN_TYPE_ENABLED;
        case FlashAttnMode::disabled:
            return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    throw BridgeError("internal_error", "unknown flash attention mode");
}

RequestConfig parse_config(const Json::Object & object) {
    const Json::Object & runtime = as_object(required_field(object, "runtime"), "runtime");
    require_exact_keys(
        runtime,
        {
            "n_ctx",
            "n_batch",
            "n_gpu_layers",
            "flash_attn",
            "seed",
            "n_threads",
            "n_threads_batch",
            "chat_template",
            "thinking_mode",
        }
    );
    RuntimeConfig runtime_config;
    runtime_config.n_ctx = required_u32(runtime, "n_ctx", 1U);
    runtime_config.n_batch = required_u32(runtime, "n_batch", 1U);
    runtime_config.n_gpu_layers = required_i32(runtime, "n_gpu_layers", -1);
    runtime_config.flash_attn = parse_flash_attn(runtime);
    runtime_config.seed = required_u32(runtime, "seed", 0U);
    runtime_config.n_threads = required_i32(runtime, "n_threads", 1);
    runtime_config.n_threads_batch = required_i32(runtime, "n_threads_batch", 1);
    runtime_config.chat_template = required_string(runtime, "chat_template");
    runtime_config.thinking_mode = required_string(runtime, "thinking_mode");
    if (runtime_config.thinking_mode != "disabled") {
        throw BridgeError("invalid_request", "runtime.thinking_mode must be disabled");
    }
    if (runtime_config.n_batch > runtime_config.n_ctx) {
        throw BridgeError("invalid_request", "runtime.n_batch must not exceed runtime.n_ctx");
    }
    return {
        required_string(object, "model_artifact"),
        required_string(object, "llama_cpp_library"),
        required_string(object, "llama_cpp_revision"),
        std::move(runtime_config),
    };
}

class Batch final {
  public:
    explicit Batch(std::size_t capacity) {
        if (capacity == 0U ||
            capacity > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw BridgeError("invalid_request", "batch capacity is outside the supported range");
        }
        batch_ = llama_batch_init(static_cast<std::int32_t>(capacity), 0, 1);
        initialized_ = true;
        if (batch_.token == nullptr || batch_.pos == nullptr || batch_.n_seq_id == nullptr ||
            batch_.seq_id == nullptr || batch_.logits == nullptr) {
            throw BridgeError("backend_unavailable", "llama.cpp could not allocate a token batch");
        }
    }

    Batch(const Batch &) = delete;
    Batch & operator=(const Batch &) = delete;

    ~Batch() {
        if (initialized_) {
            llama_batch_free(batch_);
        }
    }

    llama_batch & raw() {
        return batch_;
    }

  private:
    llama_batch batch_{};
    bool initialized_ = false;
};

class Context final {
  public:
    explicit Context(llama_context * raw) : raw_(raw) {
        if (raw_ == nullptr) {
            throw BridgeError("backend_unavailable", "llama.cpp could not create a context");
        }
    }

    Context(const Context &) = delete;
    Context & operator=(const Context &) = delete;

    Context(Context && other) noexcept : raw_(std::exchange(other.raw_, nullptr)) {}

    Context & operator=(Context && other) noexcept {
        if (this != &other) {
            if (raw_ != nullptr) {
                llama_free(raw_);
            }
            raw_ = std::exchange(other.raw_, nullptr);
        }
        return *this;
    }

    ~Context() {
        if (raw_ != nullptr) {
            llama_free(raw_);
        }
    }

    llama_context * raw() const {
        return raw_;
    }

  private:
    llama_context * raw_ = nullptr;
};

class ModelRuntime final {
  public:
    explicit ModelRuntime(const RequestConfig & config) : config_(config) {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = config_.runtime.n_gpu_layers;
        model_ = llama_model_load_from_file(config_.model_artifact.c_str(), model_params);
        if (model_ == nullptr) {
            throw BridgeError("backend_unavailable", "llama.cpp could not load the requested model artifact");
        }
        vocab_ = llama_model_get_vocab(model_);
        if (vocab_ == nullptr) {
            throw BridgeError("backend_unavailable", "llama.cpp model has no vocabulary");
        }
        vocabulary_size_ = llama_vocab_n_tokens(vocab_);
        if (vocabulary_size_ <= 0) {
            throw BridgeError("backend_unavailable", "llama.cpp model has an invalid vocabulary");
        }
    }

    ModelRuntime(const ModelRuntime &) = delete;
    ModelRuntime & operator=(const ModelRuntime &) = delete;

    ~ModelRuntime() {
        if (model_ != nullptr) {
            llama_model_free(model_);
        }
    }

    Context make_context() const {
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = config_.runtime.n_ctx;
        context_params.n_batch = config_.runtime.n_batch;
        context_params.n_threads = config_.runtime.n_threads;
        context_params.n_threads_batch = config_.runtime.n_threads_batch;
        context_params.flash_attn_type = flash_attn_type(config_.runtime.flash_attn);
        return Context(llama_init_from_model(model_, context_params));
    }

    const RequestConfig & config() const {
        return config_;
    }

    std::int32_t vocabulary_size() const {
        return vocabulary_size_;
    }

    llama_token deterministic_probe_token() const {
        const llama_token bos = llama_vocab_bos(vocab_);
        return bos >= 0 && bos < vocabulary_size_ ? bos : 0;
    }

    void validate_tokens(const std::vector<llama_token> & tokens, const char * field) const {
        for (const llama_token token : tokens) {
            if (token < 0 || token >= vocabulary_size_) {
                throw BridgeError("invalid_request", std::string(field) + " contains a token outside the model vocabulary");
            }
        }
    }

  private:
    RequestConfig config_;
    llama_model * model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    std::int32_t vocabulary_size_ = 0;
};

struct SeedSpec {
    std::string seed_id;
    std::vector<llama_token> initial_context_tokens;
    std::uint32_t steps = 0U;
};

SeedSpec parse_seed(const Json::Object & request, bool require_steps) {
    const Json::Object & seed = as_object(required_field(request, "seed"), "seed");
    if (require_steps) {
        require_exact_keys(seed, {"seed_id", "initial_context_tokens", "steps"});
    } else {
        require_exact_keys(seed, {"seed_id", "initial_context_tokens"});
    }
    SeedSpec result;
    result.seed_id = required_string(seed, "seed_id");
    result.initial_context_tokens = required_tokens(seed, "initial_context_tokens");
    if (require_steps) {
        result.steps = required_u32(seed, "steps", 0U);
    }
    return result;
}

Json runtime_echo(const RequestConfig & config) {
    Json::Object runtime;
    runtime["n_ctx"] = Json(static_cast<std::int64_t>(config.runtime.n_ctx));
    runtime["n_batch"] = Json(static_cast<std::int64_t>(config.runtime.n_batch));
    runtime["n_gpu_layers"] = Json(static_cast<std::int64_t>(config.runtime.n_gpu_layers));
    runtime["flash_attn"] = Json(flash_attn_name(config.runtime.flash_attn));
    runtime["seed"] = Json(static_cast<std::int64_t>(config.runtime.seed));
    runtime["n_threads"] = Json(static_cast<std::int64_t>(config.runtime.n_threads));
    runtime["n_threads_batch"] = Json(static_cast<std::int64_t>(config.runtime.n_threads_batch));
    runtime["chat_template"] = Json(config.runtime.chat_template);
    runtime["thinking_mode"] = Json(config.runtime.thinking_mode);

    Json::Object result;
    result["model_artifact"] = Json(config.model_artifact);
    result["llama_cpp_library"] = Json(config.declared_llama_cpp_library);
    result["llama_cpp_revision"] = Json(config.declared_llama_cpp_revision);
    result["runtime"] = Json(std::move(runtime));
    return Json(std::move(result));
}

Json observed_context(
    const Context & context
) {
    Json::Object result;
    result["actual_n_ctx"] = Json(static_cast<std::int64_t>(llama_n_ctx(context.raw())));
    result["actual_n_batch"] = Json(static_cast<std::int64_t>(llama_n_batch(context.raw())));
    result["actual_n_threads"] = Json(static_cast<std::int64_t>(llama_n_threads(context.raw())));
    result["actual_n_threads_batch"] = Json(static_cast<std::int64_t>(llama_n_threads_batch(context.raw())));
    return Json(std::move(result));
}

void ensure_context_capacity(
    const Context & context,
    std::size_t position,
    std::size_t token_count
) {
    const std::uint32_t actual_context = llama_n_ctx(context.raw());
    if (actual_context == 0U || position > static_cast<std::size_t>(actual_context) ||
        token_count > static_cast<std::size_t>(actual_context) - position) {
        throw BridgeError("context_overflow", "requested tokens exceed the context created by llama.cpp");
    }
}

void decode_tokens(
    Context & context,
    const std::vector<llama_token> & tokens,
    std::size_t start_position,
    bool single_batch
) {
    if (tokens.empty()) {
        throw BridgeError("internal_error", "decode requires at least one token");
    }
    ensure_context_capacity(context, start_position, tokens.size());
    const std::uint32_t actual_batch = llama_n_batch(context.raw());
    if (actual_batch == 0U) {
        throw BridgeError("backend_unavailable", "llama.cpp context reports n_batch=0");
    }
    if (single_batch && tokens.size() > static_cast<std::size_t>(actual_batch)) {
        throw BridgeError("invalid_request", "batched reconstruction exceeds the actual llama.cpp n_batch");
    }

    std::size_t offset = 0U;
    while (offset < tokens.size()) {
        const std::size_t count = single_batch
            ? tokens.size()
            : std::min(tokens.size() - offset, static_cast<std::size_t>(actual_batch));
        Batch batch(count);
        llama_batch & raw = batch.raw();
        raw.n_tokens = static_cast<std::int32_t>(count);
        for (std::size_t index = 0; index < count; ++index) {
            raw.token[index] = tokens[offset + index];
            raw.pos[index] = static_cast<llama_pos>(start_position + offset + index);
            raw.n_seq_id[index] = 1;
            raw.seq_id[index][0] = 0;
            raw.logits[index] = (offset + index + 1U == tokens.size()) ? 1 : 0;
        }
        const std::int32_t decode_status = llama_decode(context.raw(), raw);
        if (decode_status != 0) {
            throw BridgeError("decode_failed", "llama.cpp rejected the requested decode batch");
        }
        offset += count;
    }
}

llama_token greedy_argmax_at(
    const Context & context,
    std::int32_t vocabulary_size,
    std::int32_t output_index
) {
    float * logits = llama_get_logits_ith(context.raw(), output_index);
    if (logits == nullptr) {
        throw BridgeError("logits_unavailable", "llama.cpp did not return logits for the final token");
    }
    llama_token best_token = 0;
    float best_logit = logits[0];
    if (!std::isfinite(best_logit)) {
        throw BridgeError("logits_unavailable", "llama.cpp returned a non-finite logit");
    }
    for (llama_token token = 1; token < vocabulary_size; ++token) {
        const float candidate = logits[token];
        if (!std::isfinite(candidate)) {
            throw BridgeError("logits_unavailable", "llama.cpp returned a non-finite logit");
        }
        if (candidate > best_logit) {
            best_logit = candidate;
            best_token = token;
        }
    }
    return best_token;
}

llama_token greedy_argmax(const Context & context, std::int32_t vocabulary_size) {
    return greedy_argmax_at(context, vocabulary_size, -1);
}

void decode_one_batch_with_all_logits(
    Context & context,
    const std::vector<llama_token> & tokens,
    std::size_t start_position
) {
    if (tokens.empty()) {
        throw BridgeError("internal_error", "verification batch requires at least one token");
    }
    ensure_context_capacity(context, start_position, tokens.size());
    const std::uint32_t actual_batch = llama_n_batch(context.raw());
    if (actual_batch == 0U || tokens.size() > static_cast<std::size_t>(actual_batch)) {
        throw BridgeError("invalid_request", "verification batch exceeds the actual llama.cpp n_batch");
    }

    Batch batch(tokens.size());
    llama_batch & raw = batch.raw();
    raw.n_tokens = static_cast<std::int32_t>(tokens.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        raw.token[index] = tokens[index];
        raw.pos[index] = static_cast<llama_pos>(start_position + index);
        raw.n_seq_id[index] = 1;
        raw.seq_id[index][0] = 0;
        raw.logits[index] = 1;
    }
    if (llama_decode(context.raw(), raw) != 0) {
        throw BridgeError("decode_failed", "llama.cpp rejected the verification batch");
    }
}

std::vector<std::uint8_t> snapshot_state(const Context & context) {
    llama_synchronize(context.raw());
    const std::size_t state_size = llama_state_get_size(context.raw());
    if (state_size == 0U) {
        throw BridgeError("state_export_failed", "llama.cpp reported an empty model-memory state");
    }
    std::vector<std::uint8_t> state(state_size);
    const std::size_t copied = llama_state_get_data(context.raw(), state.data(), state.size());
    if (copied != state.size()) {
        throw BridgeError("state_export_failed", "llama.cpp did not export the complete model-memory state");
    }
    return state;
}

void restore_state(Context & context, const std::vector<std::uint8_t> & state) {
    if (state.empty()) {
        throw BridgeError("state_restore_failed", "refusing to restore an empty model-memory state");
    }
    const std::size_t restored = llama_state_set_data(context.raw(), state.data(), state.size());
    if (restored != state.size()) {
        throw BridgeError("state_restore_failed", "llama.cpp did not restore the complete model-memory state");
    }
}

std::vector<llama_token> serial_greedy_tokens(
    Context & context,
    std::size_t starting_position,
    std::size_t count,
    std::int32_t vocabulary_size
) {
    std::vector<llama_token> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const llama_token token = greedy_argmax(context, vocabulary_size);
        result.push_back(token);
        const std::vector<llama_token> one_token{token};
        decode_tokens(context, one_token, starting_position + index, true);
    }
    return result;
}

Json::Object result_base(
    const char * schema,
    const char * operation,
    const char * status
) {
    Json::Object result;
    result["schema_version"] = Json(schema);
    result["operation"] = Json(operation);
    result["status"] = Json(status);
    result["backend"] = Json("llama.cpp");
    return result;
}

std::string loaded_llama_cpp_library_path() {
#if defined(__APPLE__) || defined(__linux__)
    Dl_info information{};
    const auto function_address = reinterpret_cast<const void *>(
        reinterpret_cast<std::uintptr_t>(&llama_backend_init)
    );
    if (dladdr(function_address, &information) == 0 || information.dli_fname == nullptr ||
        information.dli_fname[0] == '\0') {
        throw BridgeError("backend_unavailable", "could not identify the loaded llama.cpp library");
    }

    // `dladdr` normally reports an absolute dynamic-loader path. Resolve a
    // symlink when possible so the caller can hash the actual loaded file.
    char * const canonical_path = realpath(information.dli_fname, nullptr);
    if (canonical_path != nullptr) {
        const std::string result(canonical_path);
        std::free(canonical_path);
        return result;
    }
    return std::string(information.dli_fname);
#else
    throw BridgeError(
        "backend_unavailable",
        "loaded llama.cpp library identity is unsupported on this platform"
    );
#endif
}

struct HandledResponse {
    Json payload;
    bool successful = false;
};

struct CapabilityProbe {
    llama_token token = 0;
    std::vector<std::uint8_t> exported_state;
    std::vector<std::uint8_t> restored_state;
    std::int32_t vocabulary_size = 0;
    std::uint32_t actual_n_ctx = 0U;
    std::uint32_t actual_n_batch = 0U;
    std::int32_t actual_n_threads = 0;
    std::int32_t actual_n_threads_batch = 0;
};

CapabilityProbe probe_configured_backend(const RequestConfig & config) {
    ModelRuntime runtime(config);
    Context source = runtime.make_context();
    const llama_token token = runtime.deterministic_probe_token();
    const std::vector<llama_token> one_token{token};
    decode_tokens(source, one_token, 0U, true);
    const std::vector<std::uint8_t> exported_state = snapshot_state(source);

    Context restored = runtime.make_context();
    restore_state(restored, exported_state);
    const std::vector<std::uint8_t> restored_state = snapshot_state(restored);
    if (exported_state != restored_state) {
        throw BridgeError(
            "state_restore_failed",
            "llama.cpp model-memory state did not survive a byte-exact export/restore round trip"
        );
    }
    return {
        token,
        exported_state,
        restored_state,
        runtime.vocabulary_size(),
        llama_n_ctx(source.raw()),
        llama_n_batch(source.raw()),
        llama_n_threads(source.raw()),
        llama_n_threads_batch(source.raw()),
    };
}

HandledResponse capabilities(const Json::Object & request) {
    require_exact_keys(
        request,
        {
            "schema_version",
            "operation",
            "model_artifact",
            "llama_cpp_library",
            "llama_cpp_revision",
            "runtime",
        }
    );
    require_request_schema(request, "capabilities");
    const RequestConfig config = parse_config(request);
    const CapabilityProbe probe = probe_configured_backend(config);

    Json::Object result = result_base(kCapabilitiesSchema, "capabilities", "ok");
    result["state_export"] = Json(kStateExport);
    result["generation_mode"] = Json("greedy_argmax");
    result["greedy_tie_break"] = Json("lowest-token-id/v1");
    result["raw_token_ids"] = Json(true);
    result["llama_c_api_version"] = Json("not-exposed-by-stable-c-api/v1");
    result["loaded_llama_cpp_library_path"] = Json(loaded_llama_cpp_library_path());
    result["supports_gpu_offload"] = Json(llama_supports_gpu_offload());
    result["supports_mmap"] = Json(llama_supports_mmap());
    result["supports_mlock"] = Json(llama_supports_mlock());
    result["vocabulary_size"] = Json(static_cast<std::int64_t>(probe.vocabulary_size));
    result["configuration"] = runtime_echo(config);
    result["observed_context"] = Json(Json::Object{
        {"actual_n_ctx", Json(static_cast<std::int64_t>(probe.actual_n_ctx))},
        {"actual_n_batch", Json(static_cast<std::int64_t>(probe.actual_n_batch))},
        {"actual_n_threads", Json(static_cast<std::int64_t>(probe.actual_n_threads))},
        {"actual_n_threads_batch", Json(static_cast<std::int64_t>(probe.actual_n_threads_batch))},
    });
    result["state_components"] = Json(Json::Array{
        Json("model_architecture_identity"),
        Json("llama_memory"),
    });
    result["state_api"] = Json(kStateApi);
    result["declared_identity_fields"] = Json("caller-declared-not-verified-by-bridge/v1");
    result["chat_template_applied"] = Json(false);
    result["thinking_mode_applied"] = Json(false);
    result["sampler_rng_state"] = Json("not-applicable-no-sampler-or-rng-in-greedy-bridge/v1");
    result["probe_token"] = Json(static_cast<std::int64_t>(probe.token));
    result["probe_state_sha256"] = Json(sha256_hex(probe.exported_state));
    result["probe_state_bytes"] = Json(static_cast<std::int64_t>(probe.exported_state.size()));
    result["restored_state_sha256"] = Json(sha256_hex(probe.restored_state));
    result["restored_state_bytes"] = Json(static_cast<std::int64_t>(probe.restored_state.size()));
    result["state_round_trip_byte_exact"] = Json(true);
    result["operations"] = Json(Json::Array{
        Json("capabilities"),
        Json("reference"),
        Json("reference_pending"),
        Json("verify"),
        Json("verify_pending"),
        Json("trajectory_pending"),
        Json("benchmark_pending"),
        Json("profile"),
        Json("profile_pending"),
    });
    return {Json(std::move(result)), true};
}

HandledResponse reference(const Json::Object & request, bool pending_protocol = false) {
    require_exact_keys(
        request,
        {
            "schema_version",
            "operation",
            "model_artifact",
            "llama_cpp_library",
            "llama_cpp_revision",
            "runtime",
            "seed",
            "reference",
        }
    );
    const char * operation = pending_protocol ? "reference_pending" : "reference";
    require_request_schema(request, operation);
    const RequestConfig config = parse_config(request);
    const SeedSpec seed = parse_seed(request, true);
    const Json::Object & reference_config = as_object(required_field(request, "reference"), "reference");
    require_exact_keys(reference_config, {"method"});
    const char * reference_method = pending_protocol
        ? "pending-token-serial-greedy-memory-state/v1"
        : "serial-greedy-memory-state/v2";
    if (required_string(reference_config, "method") != reference_method) {
        throw BridgeError("invalid_request", "reference.method is unsupported");
    }
    if (seed.initial_context_tokens.size() > config.runtime.n_ctx ||
        static_cast<std::size_t>(seed.steps) >
            static_cast<std::size_t>(config.runtime.n_ctx) - seed.initial_context_tokens.size()) {
        throw BridgeError("invalid_request", "steps exceed the remaining n_ctx");
    }

    ModelRuntime runtime(config);
    runtime.validate_tokens(seed.initial_context_tokens, "seed.initial_context_tokens");
    Context source = runtime.make_context();
    if (pending_protocol) {
        const std::vector<llama_token> prefix_tokens(
            seed.initial_context_tokens.begin(),
            seed.initial_context_tokens.end() - 1
        );
        if (!prefix_tokens.empty()) {
            decode_tokens(source, prefix_tokens, 0U, false);
        }
        decode_tokens(
            source,
            std::vector<llama_token>{seed.initial_context_tokens.back()},
            seed.initial_context_tokens.size() - 1U,
            true
        );
    } else {
        decode_tokens(source, seed.initial_context_tokens, 0U, false);
    }
    const std::vector<std::uint8_t> prompt_state = snapshot_state(source);
    const std::vector<llama_token> target_tokens = serial_greedy_tokens(
        source,
        seed.initial_context_tokens.size(),
        seed.steps,
        runtime.vocabulary_size()
    );
    const std::vector<std::uint8_t> completion_state = snapshot_state(source);

    Json::Object result = result_base(
        pending_protocol ? kPendingReferenceSchema : kReferenceSchema,
        operation,
        "ok"
    );
    result["generation_mode"] = Json("greedy_argmax");
    result["greedy_tie_break"] = Json("lowest-token-id/v1");
    result["seed_id"] = Json(seed.seed_id);
    result["target_tokens"] = token_array(target_tokens);
    result["vocabulary_size"] = Json(static_cast<std::int64_t>(runtime.vocabulary_size()));
    result["state_export"] = Json(kStateExport);
    result["prompt_state_sha256"] = Json(sha256_hex(prompt_state));
    result["prompt_state_bytes"] = Json(static_cast<std::int64_t>(prompt_state.size()));
    result["completion_state_sha256"] = Json(sha256_hex(completion_state));
    result["completion_state_bytes"] = Json(static_cast<std::int64_t>(completion_state.size()));
    result["mutable_state_digest"] = Json(sha256_hex(completion_state));
    result["configuration"] = runtime_echo(config);
    result["observed_context"] = observed_context(source);
    result["tokenization_boundary"] = Json("caller-supplied-raw-token-ids/v1");
    result["state_api"] = Json(kStateApi);
    result["declared_identity_fields"] = Json("caller-declared-not-verified-by-bridge/v1");
    result["chat_template_applied"] = Json(false);
    result["thinking_mode_applied"] = Json(false);
    result["sampler_rng_state"] = Json("not-applicable-no-sampler-or-rng-in-greedy-bridge/v1");
    return {Json(std::move(result)), true};
}

HandledResponse verify(const Json::Object & request) {
    require_exact_keys(
        request,
        {
            "schema_version",
            "operation",
            "model_artifact",
            "llama_cpp_library",
            "llama_cpp_revision",
            "runtime",
            "seed",
            "case",
        }
    );
    require_request_schema(request, "verify");
    const RequestConfig config = parse_config(request);
    const SeedSpec seed = parse_seed(request, false);
    const Json::Object & case_config = as_object(required_field(request, "case"), "case");
    require_exact_keys(case_config, {"case_id", "draft_tokens", "include_bonus_on_full_accept"});
    const std::string case_id = required_string(case_config, "case_id");
    const std::vector<llama_token> draft_tokens = required_tokens(case_config, "draft_tokens");
    const bool include_bonus = required_boolean(case_config, "include_bonus_on_full_accept");
    const std::size_t target_count = draft_tokens.size() + 1U;
    if (seed.initial_context_tokens.size() > config.runtime.n_ctx ||
        target_count > static_cast<std::size_t>(config.runtime.n_ctx) - seed.initial_context_tokens.size()) {
        throw BridgeError("invalid_request", "draft_tokens plus requested bonus exceed the remaining n_ctx");
    }

    ModelRuntime runtime(config);
    runtime.validate_tokens(seed.initial_context_tokens, "seed.initial_context_tokens");
    runtime.validate_tokens(draft_tokens, "draft_tokens");
    Context source = runtime.make_context();
    decode_tokens(source, seed.initial_context_tokens, 0U, false);
    const std::vector<std::uint8_t> prompt_state = snapshot_state(source);
    const llama_token prompt_target = greedy_argmax(source, runtime.vocabulary_size());
    const std::vector<llama_token> target_tokens = serial_greedy_tokens(
        source,
        seed.initial_context_tokens.size(),
        target_count,
        runtime.vocabulary_size()
    );
    if (target_tokens.front() != prompt_target) {
        throw BridgeError("internal_error", "serial reference changed before its first target token");
    }

    Context verifier = runtime.make_context();
    restore_state(verifier, prompt_state);
    decode_one_batch_with_all_logits(verifier, draft_tokens, seed.initial_context_tokens.size());
    std::vector<llama_token> verification_target_tokens;
    verification_target_tokens.reserve(target_count);
    verification_target_tokens.push_back(prompt_target);
    for (std::size_t index = 0; index < draft_tokens.size(); ++index) {
        verification_target_tokens.push_back(greedy_argmax_at(
            verifier,
            runtime.vocabulary_size(),
            static_cast<std::int32_t>(index)
        ));
    }
    const std::vector<std::uint8_t> verifier_draft_state = snapshot_state(verifier);

    std::size_t accepted = 0U;
    while (accepted < draft_tokens.size() &&
           draft_tokens[accepted] == verification_target_tokens[accepted]) {
        ++accepted;
    }
    std::vector<llama_token> committed_tokens(
        draft_tokens.begin(),
        draft_tokens.begin() + static_cast<std::ptrdiff_t>(accepted)
    );
    Json correction(nullptr);
    Json bonus(nullptr);
    if (accepted < draft_tokens.size()) {
        const llama_token correction_token = verification_target_tokens[accepted];
        committed_tokens.push_back(correction_token);
        correction = Json(static_cast<std::int64_t>(correction_token));
    } else if (include_bonus) {
        const llama_token bonus_token = verification_target_tokens[draft_tokens.size()];
        committed_tokens.push_back(bonus_token);
        bonus = Json(static_cast<std::int64_t>(bonus_token));
    }
    if (committed_tokens.empty()) {
        throw BridgeError("internal_error", "verification produced no committed token");
    }

    std::size_t reference_accepted = 0U;
    while (reference_accepted < draft_tokens.size() &&
           draft_tokens[reference_accepted] == target_tokens[reference_accepted]) {
        ++reference_accepted;
    }
    std::vector<llama_token> reference_committed_tokens(
        draft_tokens.begin(),
        draft_tokens.begin() + static_cast<std::ptrdiff_t>(reference_accepted)
    );
    if (reference_accepted < draft_tokens.size()) {
        reference_committed_tokens.push_back(target_tokens[reference_accepted]);
    } else if (include_bonus) {
        reference_committed_tokens.push_back(target_tokens[draft_tokens.size()]);
    }

    const std::size_t authoritative_target_count = accepted < draft_tokens.size()
        ? accepted + 1U
        : verification_target_tokens.size();
    const bool verification_targets_match = std::equal(
        verification_target_tokens.begin(),
        verification_target_tokens.begin() + static_cast<std::ptrdiff_t>(authoritative_target_count),
        target_tokens.begin()
    );

    Context reference = runtime.make_context();
    restore_state(reference, prompt_state);
    for (std::size_t index = 0; index < reference_committed_tokens.size(); ++index) {
        const std::vector<llama_token> one_token{reference_committed_tokens[index]};
        decode_tokens(reference, one_token, seed.initial_context_tokens.size() + index, true);
    }
    const llama_token reference_next = greedy_argmax(reference, runtime.vocabulary_size());
    const std::vector<std::uint8_t> reference_state = snapshot_state(reference);

    // The verifier batch is disposable. Reconstruct the committed path serially
    // from the prompt snapshot so rollback never retains rejected or numerically
    // batch-dependent memory state.
    Context committed = runtime.make_context();
    restore_state(committed, prompt_state);
    for (std::size_t index = 0; index < committed_tokens.size(); ++index) {
        const std::vector<llama_token> one_token{committed_tokens[index]};
        decode_tokens(committed, one_token, seed.initial_context_tokens.size() + index, true);
    }
    const llama_token committed_next = greedy_argmax(committed, runtime.vocabulary_size());
    const std::vector<std::uint8_t> committed_state = snapshot_state(committed);

    const bool state_matches = reference_state == committed_state;
    const bool continuation_matches = reference_next == committed_next;
    const bool successful = verification_targets_match && state_matches && continuation_matches;
    Json::Object result = result_base(
        kVerifySchema,
        "verify",
        successful ? "ok" : "mismatch"
    );
    result["generation_mode"] = Json("greedy_argmax");
    result["greedy_tie_break"] = Json("lowest-token-id/v1");
    result["state_export"] = Json(kStateExport);
    result["reconstruction"] = Json("batch-score+serial-canonical-commit/v2");
    result["seed_id"] = Json(seed.seed_id);
    result["case_id"] = Json(case_id);
    result["target_tokens"] = token_array(target_tokens);
    result["verification_target_tokens"] = token_array(verification_target_tokens);
    result["verification_targets_match"] = Json(verification_targets_match);
    result["vocabulary_size"] = Json(static_cast<std::int64_t>(runtime.vocabulary_size()));
    result["accepted_draft_tokens"] = Json(static_cast<std::int64_t>(accepted));
    result["reference_accepted_draft_tokens"] = Json(static_cast<std::int64_t>(reference_accepted));
    result["correction_token"] = std::move(correction);
    result["bonus_token"] = std::move(bonus);
    result["committed_tokens"] = token_array(committed_tokens);
    result["reference_committed_tokens"] = token_array(reference_committed_tokens);
    result["prompt_state_sha256"] = Json(sha256_hex(prompt_state));
    result["prompt_state_bytes"] = Json(static_cast<std::int64_t>(prompt_state.size()));
    result["reference_state_sha256"] = Json(sha256_hex(reference_state));
    result["reference_state_bytes"] = Json(static_cast<std::int64_t>(reference_state.size()));
    result["committed_state_sha256"] = Json(sha256_hex(committed_state));
    result["committed_state_bytes"] = Json(static_cast<std::int64_t>(committed_state.size()));
    result["verifier_draft_state_sha256"] = Json(sha256_hex(verifier_draft_state));
    result["verifier_draft_state_bytes"] = Json(static_cast<std::int64_t>(verifier_draft_state.size()));
    result["reference_mutable_state_digest"] = Json(sha256_hex(reference_state));
    result["committed_mutable_state_digest"] = Json(sha256_hex(committed_state));
    result["mutable_state_digest"] = Json(sha256_hex(committed_state));
    result["state_matches"] = Json(state_matches);
    result["reference_next_token"] = Json(static_cast<std::int64_t>(reference_next));
    result["committed_next_token"] = Json(static_cast<std::int64_t>(committed_next));
    result["continuation_matches"] = Json(continuation_matches);
    result["configuration"] = runtime_echo(config);
    result["observed_context"] = observed_context(committed);
    result["tokenization_boundary"] = Json("caller-supplied-raw-token-ids/v1");
    result["state_api"] = Json(kStateApi);
    result["declared_identity_fields"] = Json("caller-declared-not-verified-by-bridge/v1");
    result["chat_template_applied"] = Json(false);
    result["thinking_mode_applied"] = Json(false);
    result["sampler_rng_state"] = Json("not-applicable-no-sampler-or-rng-in-greedy-bridge/v1");
    return {Json(std::move(result)), successful};
}

HandledResponse verify_pending(const Json::Object & request) {
    require_exact_keys(
        request,
        {
            "schema_version",
            "operation",
            "model_artifact",
            "llama_cpp_library",
            "llama_cpp_revision",
            "runtime",
            "seed",
            "case",
        }
    );
    require_request_schema(request, "verify_pending");
    const RequestConfig config = parse_config(request);
    const SeedSpec seed = parse_seed(request, false);
    const Json::Object & case_config = as_object(required_field(request, "case"), "case");
    require_exact_keys(case_config, {"case_id", "draft_tokens", "include_bonus_on_full_accept"});
    const std::string case_id = required_string(case_config, "case_id");
    const std::vector<llama_token> draft_tokens = required_tokens(case_config, "draft_tokens");
    const bool include_bonus = required_boolean(case_config, "include_bonus_on_full_accept");
    const std::size_t target_count = draft_tokens.size() + 1U;
    if (seed.initial_context_tokens.size() > config.runtime.n_ctx ||
        target_count > static_cast<std::size_t>(config.runtime.n_ctx) - seed.initial_context_tokens.size()) {
        throw BridgeError("invalid_request", "draft_tokens plus requested bonus exceed the remaining n_ctx");
    }
    if (draft_tokens.size() + 1U > config.runtime.n_batch) {
        throw BridgeError("invalid_request", "pending-token verification batch exceeds configured n_batch");
    }

    ModelRuntime runtime(config);
    runtime.validate_tokens(seed.initial_context_tokens, "seed.initial_context_tokens");
    runtime.validate_tokens(draft_tokens, "draft_tokens");
    const std::size_t prefix_size = seed.initial_context_tokens.size() - 1U;
    const llama_token anchor_token = seed.initial_context_tokens.back();
    const std::vector<llama_token> prefix_tokens(
        seed.initial_context_tokens.begin(),
        seed.initial_context_tokens.end() - 1
    );
    auto decode_prefix = [&](Context & context) {
        if (!prefix_tokens.empty()) {
            decode_tokens(context, prefix_tokens, 0U, false);
        }
    };

    Context serial_target = runtime.make_context();
    decode_prefix(serial_target);
    std::vector<llama_token> target_tokens;
    target_tokens.reserve(target_count);
    llama_token pending_target = anchor_token;
    for (std::size_t index = 0; index < target_count; ++index) {
        decode_tokens(serial_target, std::vector<llama_token>{pending_target}, prefix_size + index, true);
        pending_target = greedy_argmax(serial_target, runtime.vocabulary_size());
        target_tokens.push_back(pending_target);
    }

    Context verifier = runtime.make_context();
    decode_prefix(verifier);
    const std::vector<std::uint8_t> prompt_state = snapshot_state(verifier);
    std::vector<llama_token> verification_batch;
    verification_batch.reserve(draft_tokens.size() + 1U);
    verification_batch.push_back(anchor_token);
    verification_batch.insert(verification_batch.end(), draft_tokens.begin(), draft_tokens.end());
    decode_one_batch_with_all_logits(verifier, verification_batch, prefix_size);
    std::vector<llama_token> verification_target_tokens;
    verification_target_tokens.reserve(target_count);
    for (std::size_t index = 0; index < verification_batch.size(); ++index) {
        verification_target_tokens.push_back(greedy_argmax_at(
            verifier,
            runtime.vocabulary_size(),
            static_cast<std::int32_t>(index)
        ));
    }
    const std::vector<std::uint8_t> verifier_draft_state = snapshot_state(verifier);

    std::size_t accepted = 0U;
    while (accepted < draft_tokens.size() &&
           draft_tokens[accepted] == verification_target_tokens[accepted]) {
        ++accepted;
    }
    std::vector<llama_token> committed_tokens(
        draft_tokens.begin(),
        draft_tokens.begin() + static_cast<std::ptrdiff_t>(accepted)
    );
    Json correction(nullptr);
    Json bonus(nullptr);
    bool has_pending_commit = false;
    llama_token pending_commit = 0;
    if (accepted < draft_tokens.size()) {
        pending_commit = verification_target_tokens[accepted];
        committed_tokens.push_back(pending_commit);
        correction = Json(static_cast<std::int64_t>(pending_commit));
        has_pending_commit = true;
    } else if (include_bonus) {
        pending_commit = verification_target_tokens[draft_tokens.size()];
        committed_tokens.push_back(pending_commit);
        bonus = Json(static_cast<std::int64_t>(pending_commit));
        has_pending_commit = true;
    }
    if (committed_tokens.empty()) {
        throw BridgeError("internal_error", "pending-token verification produced no committed token");
    }

    std::size_t reference_accepted = 0U;
    while (reference_accepted < draft_tokens.size() &&
           draft_tokens[reference_accepted] == target_tokens[reference_accepted]) {
        ++reference_accepted;
    }
    std::vector<llama_token> reference_committed_tokens(
        draft_tokens.begin(),
        draft_tokens.begin() + static_cast<std::ptrdiff_t>(reference_accepted)
    );
    if (reference_accepted < draft_tokens.size()) {
        reference_committed_tokens.push_back(target_tokens[reference_accepted]);
    } else if (include_bonus) {
        reference_committed_tokens.push_back(target_tokens[draft_tokens.size()]);
    }

    const std::size_t authoritative_target_count = accepted < draft_tokens.size()
        ? accepted + 1U
        : verification_target_tokens.size();
    const bool verification_targets_match = std::equal(
        verification_target_tokens.begin(),
        verification_target_tokens.begin() + static_cast<std::ptrdiff_t>(authoritative_target_count),
        target_tokens.begin()
    );

    llama_synchronize(verifier.raw());
    bool tail_trimmed = false;
    std::string reconstruction = "pending-anchor+batch-score+native-tail-trim/v1";
    if (accepted > 2U) {
        restore_state(verifier, prompt_state);
        decode_tokens(verifier, std::vector<llama_token>{anchor_token}, prefix_size, true);
        for (std::size_t index = 0; index < accepted; ++index) {
            decode_tokens(
                verifier,
                std::vector<llama_token>{draft_tokens[index]},
                prefix_size + 1U + index,
                true
            );
        }
        reconstruction = "pending-anchor+batch-score+serial-state-fallback/v2";
    } else {
        const llama_pos trim_from = static_cast<llama_pos>(prefix_size + 1U + accepted);
        tail_trimmed = llama_memory_seq_rm(llama_get_memory(verifier.raw()), 0, trim_from, -1);
        if (!tail_trimmed) {
            throw BridgeError("state_trim_unsupported", "llama.cpp cannot remove the rejected verification tail");
        }
    }
    const std::vector<std::uint8_t> pending_state = snapshot_state(verifier);

    Context pending_reference = runtime.make_context();
    decode_prefix(pending_reference);
    decode_tokens(pending_reference, std::vector<llama_token>{anchor_token}, prefix_size, true);
    for (std::size_t index = 0; index < reference_accepted; ++index) {
        decode_tokens(
            pending_reference,
            std::vector<llama_token>{draft_tokens[index]},
            prefix_size + 1U + index,
            true
        );
    }
    const std::vector<std::uint8_t> reference_pending_state = snapshot_state(pending_reference);
    const bool pending_state_matches = pending_state == reference_pending_state;

    if (has_pending_commit) {
        const std::size_t pending_position = prefix_size + 1U + accepted;
        decode_tokens(verifier, std::vector<llama_token>{pending_commit}, pending_position, true);
        decode_tokens(pending_reference, std::vector<llama_token>{pending_commit}, pending_position, true);
    }
    const llama_token committed_next = greedy_argmax(verifier, runtime.vocabulary_size());
    const std::vector<std::uint8_t> committed_state = snapshot_state(verifier);
    const llama_token reference_next = greedy_argmax(pending_reference, runtime.vocabulary_size());
    const std::vector<std::uint8_t> reference_state = snapshot_state(pending_reference);

    const bool state_matches = reference_state == committed_state;
    const bool continuation_matches = reference_next == committed_next;
    const bool successful = verification_targets_match && pending_state_matches && state_matches && continuation_matches;
    Json::Object result = result_base(
        kPendingVerifySchema,
        "verify_pending",
        successful ? "ok" : "mismatch"
    );
    result["generation_mode"] = Json("greedy_argmax");
    result["greedy_tie_break"] = Json("lowest-token-id/v1");
    result["state_export"] = Json(kStateExport);
    result["reconstruction"] = Json(reconstruction);
    result["seed_id"] = Json(seed.seed_id);
    result["case_id"] = Json(case_id);
    result["target_tokens"] = token_array(target_tokens);
    result["verification_target_tokens"] = token_array(verification_target_tokens);
    result["verification_targets_match"] = Json(verification_targets_match);
    result["vocabulary_size"] = Json(static_cast<std::int64_t>(runtime.vocabulary_size()));
    result["accepted_draft_tokens"] = Json(static_cast<std::int64_t>(accepted));
    result["reference_accepted_draft_tokens"] = Json(static_cast<std::int64_t>(reference_accepted));
    result["correction_token"] = std::move(correction);
    result["bonus_token"] = std::move(bonus);
    result["committed_tokens"] = token_array(committed_tokens);
    result["reference_committed_tokens"] = token_array(reference_committed_tokens);
    result["prompt_state_sha256"] = Json(sha256_hex(prompt_state));
    result["prompt_state_bytes"] = Json(static_cast<std::int64_t>(prompt_state.size()));
    result["reference_state_sha256"] = Json(sha256_hex(reference_state));
    result["reference_state_bytes"] = Json(static_cast<std::int64_t>(reference_state.size()));
    result["committed_state_sha256"] = Json(sha256_hex(committed_state));
    result["committed_state_bytes"] = Json(static_cast<std::int64_t>(committed_state.size()));
    result["verifier_draft_state_sha256"] = Json(sha256_hex(verifier_draft_state));
    result["verifier_draft_state_bytes"] = Json(static_cast<std::int64_t>(verifier_draft_state.size()));
    result["reference_mutable_state_digest"] = Json(sha256_hex(reference_state));
    result["committed_mutable_state_digest"] = Json(sha256_hex(committed_state));
    result["mutable_state_digest"] = Json(sha256_hex(committed_state));
    result["state_matches"] = Json(state_matches);
    result["reference_next_token"] = Json(static_cast<std::int64_t>(reference_next));
    result["committed_next_token"] = Json(static_cast<std::int64_t>(committed_next));
    result["continuation_matches"] = Json(continuation_matches);
    result["pending_state_sha256"] = Json(sha256_hex(pending_state));
    result["pending_state_bytes"] = Json(static_cast<std::int64_t>(pending_state.size()));
    result["reference_pending_state_sha256"] = Json(sha256_hex(reference_pending_state));
    result["reference_pending_state_bytes"] = Json(static_cast<std::int64_t>(reference_pending_state.size()));
    result["pending_state_matches"] = Json(pending_state_matches);
    result["tail_trimmed"] = Json(tail_trimmed);
    result["pending_token"] = has_pending_commit
        ? Json(static_cast<std::int64_t>(pending_commit))
        : Json(nullptr);
    result["configuration"] = runtime_echo(config);
    result["observed_context"] = observed_context(verifier);
    result["tokenization_boundary"] = Json("caller-supplied-raw-token-ids/v1");
    result["state_api"] = Json(kStateApi);
    result["declared_identity_fields"] = Json("caller-declared-not-verified-by-bridge/v1");
    result["chat_template_applied"] = Json(false);
    result["thinking_mode_applied"] = Json(false);
    result["sampler_rng_state"] = Json("not-applicable-no-sampler-or-rng-in-greedy-bridge/v1");
    return {Json(std::move(result)), successful};
}

HandledResponse trajectory_pending(const Json::Object & request) {
    require_exact_keys(
        request,
        {
            "schema_version",
            "operation",
            "model_artifact",
            "llama_cpp_library",
            "llama_cpp_revision",
            "runtime",
            "seed",
            "trajectory",
        }
    );
    require_request_schema(request, "trajectory_pending");
    const RequestConfig config = parse_config(request);
    const SeedSpec seed = parse_seed(request, false);
    const Json::Object & trajectory = as_object(required_field(request, "trajectory"), "trajectory");
    require_exact_keys(trajectory, {"case_id", "k", "rounds"});
    const std::string case_id = required_string(trajectory, "case_id");
    const std::size_t k = static_cast<std::size_t>(required_u32(trajectory, "k", 1U));
    const Json::Array & round_specs = as_array(required_field(trajectory, "rounds"), "trajectory.rounds");
    if (round_specs.empty() || round_specs.size() > 128U) {
        throw BridgeError("invalid_request", "trajectory.rounds must contain between 1 and 128 rounds");
    }
    if (k + 1U > config.runtime.n_batch) {
        throw BridgeError("invalid_request", "pending trajectory batch exceeds configured n_batch");
    }
    const std::size_t prefix_size = seed.initial_context_tokens.size() - 1U;
    if (prefix_size + round_specs.size() * (k + 1U) + 1U > config.runtime.n_ctx) {
        throw BridgeError("invalid_request", "pending trajectory worst-case length exceeds configured n_ctx");
    }

    ModelRuntime runtime(config);
    runtime.validate_tokens(seed.initial_context_tokens, "seed.initial_context_tokens");
    const std::vector<llama_token> prefix_tokens(
        seed.initial_context_tokens.begin(),
        seed.initial_context_tokens.end() - 1
    );
    auto decode_prefix = [&](Context & context) {
        if (!prefix_tokens.empty()) {
            decode_tokens(context, prefix_tokens, 0U, false);
        }
    };
    Context verifier = runtime.make_context();
    Context reference = runtime.make_context();
    decode_prefix(verifier);
    decode_prefix(reference);
    std::size_t committed_size = prefix_size;
    llama_token pending_token = seed.initial_context_tokens.back();
    Json::Array round_results;
    round_results.reserve(round_specs.size());
    bool all_rounds_match = true;

    for (std::size_t round_index = 0; round_index < round_specs.size(); ++round_index) {
        const Json::Object & round = as_object(round_specs[round_index], "trajectory.rounds[]");
        require_exact_keys(round, {"mode", "rejection_position"});
        const std::string mode = required_string(round, "mode");
        const std::int64_t rejection_position = required_integer(round, "rejection_position");
        const bool full_accept = mode == "full_accept";
        if ((!full_accept && mode != "reject") ||
            (full_accept && rejection_position != -1) ||
            (!full_accept && (rejection_position < 0 || static_cast<std::size_t>(rejection_position) >= k))) {
            throw BridgeError("invalid_request", "trajectory round mode/rejection_position is invalid");
        }

        const std::vector<std::uint8_t> round_start_state = snapshot_state(reference);
        Context oracle = runtime.make_context();
        restore_state(oracle, round_start_state);
        decode_tokens(oracle, std::vector<llama_token>{pending_token}, committed_size, true);
        const std::vector<llama_token> target_tokens = serial_greedy_tokens(
            oracle,
            committed_size + 1U,
            k,
            runtime.vocabulary_size()
        );
        const llama_token bonus_token = greedy_argmax(oracle, runtime.vocabulary_size());
        std::vector<llama_token> draft_tokens = target_tokens;
        std::size_t accepted = k;
        llama_token next_pending = bonus_token;
        if (!full_accept) {
            accepted = static_cast<std::size_t>(rejection_position);
            if (runtime.vocabulary_size() < 2) {
                throw BridgeError("invalid_request", "forced rejection requires at least two vocabulary tokens");
            }
            draft_tokens[accepted] = static_cast<llama_token>(
                (static_cast<std::int64_t>(draft_tokens[accepted]) + 1) % runtime.vocabulary_size()
            );
            next_pending = target_tokens[accepted];
        }

        decode_tokens(reference, std::vector<llama_token>{pending_token}, committed_size, true);
        for (std::size_t index = 0; index < accepted; ++index) {
            decode_tokens(
                reference,
                std::vector<llama_token>{target_tokens[index]},
                committed_size + 1U + index,
                true
            );
        }
        std::vector<llama_token> verification_batch;
        verification_batch.reserve(k + 1U);
        verification_batch.push_back(pending_token);
        verification_batch.insert(verification_batch.end(), draft_tokens.begin(), draft_tokens.end());
        decode_one_batch_with_all_logits(verifier, verification_batch, committed_size);
        llama_synchronize(verifier.raw());
        const llama_pos trim_from = static_cast<llama_pos>(committed_size + 1U + accepted);
        const bool tail_trimmed = llama_memory_seq_rm(llama_get_memory(verifier.raw()), 0, trim_from, -1);
        if (!tail_trimmed) {
            throw BridgeError("state_trim_unsupported", "llama.cpp cannot remove a pending trajectory tail");
        }
        const std::vector<std::uint8_t> verifier_state = snapshot_state(verifier);
        const std::vector<std::uint8_t> reference_state = snapshot_state(reference);
        const bool state_matches = verifier_state == reference_state;

        Json::Object round_result;
        round_result["round"] = Json(static_cast<std::int64_t>(round_index));
        round_result["mode"] = Json(mode);
        round_result["rejection_position"] = Json(rejection_position);
        round_result["accepted_draft_tokens"] = Json(static_cast<std::int64_t>(accepted));
        round_result["pending_token"] = Json(static_cast<std::int64_t>(next_pending));
        round_result["tail_trimmed"] = Json(tail_trimmed);
        round_result["state_matches"] = Json(state_matches);
        round_result["verifier_state_sha256"] = Json(sha256_hex(verifier_state));
        round_result["reference_state_sha256"] = Json(sha256_hex(reference_state));
        round_result["state_bytes"] = Json(static_cast<std::int64_t>(verifier_state.size()));
        round_results.emplace_back(Json(std::move(round_result)));
        if (!state_matches) {
            all_rounds_match = false;
            break;
        }
        committed_size += 1U + accepted;
        pending_token = next_pending;
    }

    bool final_state_matches = false;
    bool continuation_matches = false;
    Json final_next_token(nullptr);
    if (all_rounds_match && round_results.size() == round_specs.size()) {
        decode_tokens(verifier, std::vector<llama_token>{pending_token}, committed_size, true);
        decode_tokens(reference, std::vector<llama_token>{pending_token}, committed_size, true);
        const std::vector<std::uint8_t> verifier_final_state = snapshot_state(verifier);
        const std::vector<std::uint8_t> reference_final_state = snapshot_state(reference);
        final_state_matches = verifier_final_state == reference_final_state;
        const llama_token verifier_next = greedy_argmax(verifier, runtime.vocabulary_size());
        const llama_token reference_next = greedy_argmax(reference, runtime.vocabulary_size());
        continuation_matches = verifier_next == reference_next;
        final_next_token = Json(static_cast<std::int64_t>(verifier_next));
    }
    const bool successful = all_rounds_match && final_state_matches && continuation_matches;
    Json::Object result = result_base(
        kPendingTrajectorySchema,
        "trajectory_pending",
        successful ? "ok" : "mismatch"
    );
    result["case_id"] = Json(case_id);
    result["seed_id"] = Json(seed.seed_id);
    result["k"] = Json(static_cast<std::int64_t>(k));
    result["rounds_requested"] = Json(static_cast<std::int64_t>(round_specs.size()));
    result["rounds_completed"] = Json(static_cast<std::int64_t>(round_results.size()));
    result["rounds"] = Json(std::move(round_results));
    result["all_rounds_match"] = Json(all_rounds_match);
    result["final_state_matches"] = Json(final_state_matches);
    result["continuation_matches"] = Json(continuation_matches);
    result["final_next_token"] = std::move(final_next_token);
    result["state_export"] = Json(kStateExport);
    result["state_api"] = Json(kStateApi);
    result["configuration"] = runtime_echo(config);
    result["observed_context"] = observed_context(verifier);
    result["claim_boundary"] = Json("exact-state correctness trajectory only; no performance measurement");
    return {Json(std::move(result)), successful};
}

std::vector<std::uint32_t> required_k_values(const Json::Object & profile) {
    const Json::Array & values = as_array(required_field(profile, "k_values"), "profile.k_values");
    if (values.empty()) {
        throw BridgeError("invalid_request", "profile.k_values must not be empty");
    }
    std::vector<std::uint32_t> result;
    result.reserve(values.size());
    for (const Json & item : values) {
        if (item.kind != Json::Kind::integer || item.integer < 1 ||
            item.integer > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw BridgeError("invalid_request", "profile.k_values must contain positive u32 integers");
        }
        const auto value = static_cast<std::uint32_t>(item.integer);
        if (std::find(result.begin(), result.end(), value) != result.end()) {
            throw BridgeError("invalid_request", "profile.k_values must not contain duplicates");
        }
        result.push_back(value);
    }
    return result;
}

template <typename Function>
std::int64_t measured_nanoseconds(Function && function) {
    const auto started = std::chrono::steady_clock::now();
    function();
    const auto finished = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count();
}

HandledResponse profile(const Json::Object & request) {
    require_exact_keys(
        request,
        {
            "schema_version",
            "operation",
            "model_artifact",
            "llama_cpp_library",
            "llama_cpp_revision",
            "runtime",
            "seed",
            "profile",
        }
    );
    require_request_schema(request, "profile");
    const RequestConfig config = parse_config(request);
    const SeedSpec seed = parse_seed(request, false);
    const Json::Object & profile_config = as_object(required_field(request, "profile"), "profile");
    require_exact_keys(profile_config, {"k_values", "warmup_repetitions", "measurement_repetitions"});
    const std::vector<std::uint32_t> k_values = required_k_values(profile_config);
    const std::uint32_t warmup_repetitions = required_u32(profile_config, "warmup_repetitions", 0U);
    const std::uint32_t measurement_repetitions = required_u32(
        profile_config,
        "measurement_repetitions",
        2U
    );
    const std::uint32_t max_k = *std::max_element(k_values.begin(), k_values.end());
    if (max_k > config.runtime.n_batch) {
        throw BridgeError("invalid_request", "profile k exceeds configured n_batch");
    }
    if (seed.initial_context_tokens.size() > config.runtime.n_ctx ||
        static_cast<std::size_t>(max_k) + 1U >
            static_cast<std::size_t>(config.runtime.n_ctx) - seed.initial_context_tokens.size()) {
        throw BridgeError("invalid_request", "profile k plus target lookahead exceeds remaining n_ctx");
    }

    ModelRuntime runtime(config);
    runtime.validate_tokens(seed.initial_context_tokens, "seed.initial_context_tokens");
    Context prompt = runtime.make_context();
    decode_tokens(prompt, seed.initial_context_tokens, 0U, false);
    const std::vector<std::uint8_t> prompt_state = snapshot_state(prompt);
    const llama_token prompt_target = greedy_argmax(prompt, runtime.vocabulary_size());

    const std::vector<llama_token> perfect_tokens = serial_greedy_tokens(
        prompt,
        seed.initial_context_tokens.size(),
        static_cast<std::size_t>(max_k) + 1U,
        runtime.vocabulary_size()
    );
    if (perfect_tokens.front() != prompt_target) {
        throw BridgeError("internal_error", "profile serial target changed at the prompt boundary");
    }

    Json::Array profiles;
    for (const std::uint32_t k : k_values) {
        const std::vector<llama_token> draft(perfect_tokens.begin(), perfect_tokens.begin() + k);
        Context serial_before = runtime.make_context();
        Context verifier = runtime.make_context();
        Context serial_after = runtime.make_context();
        Context restore_before_probe = runtime.make_context();
        Context restore_batch_probe = runtime.make_context();
        Context restore_after_probe = runtime.make_context();

        auto prefill = [&](Context & context) {
            decode_tokens(context, seed.initial_context_tokens, 0U, false);
            llama_synchronize(context.raw());
        };

        auto run_serial = [&](Context & context) {
            for (std::size_t index = 0; index < draft.size(); ++index) {
                if (greedy_argmax(context, runtime.vocabulary_size()) != draft[index]) {
                    throw BridgeError("profile_correctness_failed", "serial profile target changed");
                }
                decode_tokens(
                    context,
                    std::vector<llama_token>{draft[index]},
                    seed.initial_context_tokens.size() + index,
                    true
                );
            }
            llama_synchronize(context.raw());
        };
        auto run_batch = [&]() {
            decode_one_batch_with_all_logits(verifier, draft, seed.initial_context_tokens.size());
            llama_synchronize(verifier.raw());
            for (std::size_t index = 0; index < draft.size(); ++index) {
                if (greedy_argmax_at(verifier, runtime.vocabulary_size(), static_cast<std::int32_t>(index)) !=
                    perfect_tokens[index + 1U]) {
                    throw BridgeError(
                        "profile_correctness_failed",
                        "batch verifier target differs from the serial reference"
                    );
                }
            }
        };

        for (std::uint32_t iteration = 0; iteration < warmup_repetitions; ++iteration) {
            serial_before = runtime.make_context();
            prefill(serial_before);
            run_serial(serial_before);
            verifier = runtime.make_context();
            prefill(verifier);
            run_batch();
            serial_after = runtime.make_context();
            prefill(serial_after);
            run_serial(serial_after);
        }

        Json::Array measurements;
        for (std::uint32_t iteration = 0; iteration < measurement_repetitions; ++iteration) {
            serial_before = runtime.make_context();
            prefill(serial_before);
            const std::int64_t restore_before_ns = measured_nanoseconds([&]() {
                restore_state(restore_before_probe, prompt_state);
                llama_synchronize(restore_before_probe.raw());
            });
            const std::int64_t serial_before_ns = measured_nanoseconds([&]() { run_serial(serial_before); });
            verifier = runtime.make_context();
            prefill(verifier);
            const std::int64_t restore_batch_ns = measured_nanoseconds([&]() {
                restore_state(restore_batch_probe, prompt_state);
                llama_synchronize(restore_batch_probe.raw());
            });
            const std::int64_t batch_verify_ns = measured_nanoseconds([&]() { run_batch(); });
            serial_after = runtime.make_context();
            prefill(serial_after);
            const std::int64_t restore_after_ns = measured_nanoseconds([&]() {
                restore_state(restore_after_probe, prompt_state);
                llama_synchronize(restore_after_probe.raw());
            });
            const std::int64_t serial_after_ns = measured_nanoseconds([&]() { run_serial(serial_after); });
            const std::int64_t snapshot_ns = measured_nanoseconds([&]() {
                const std::vector<std::uint8_t> state = snapshot_state(serial_after);
                if (state.empty()) {
                    throw BridgeError("internal_error", "profile snapshot unexpectedly empty");
                }
            });

            Json::Object row;
            row["iteration"] = Json(static_cast<std::int64_t>(iteration));
            row["restore_before_ns"] = Json(restore_before_ns);
            row["serial_before_ns"] = Json(serial_before_ns);
            row["restore_batch_ns"] = Json(restore_batch_ns);
            row["batch_verify_ns"] = Json(batch_verify_ns);
            row["restore_after_ns"] = Json(restore_after_ns);
            row["serial_after_ns"] = Json(serial_after_ns);
            row["snapshot_ns"] = Json(snapshot_ns);
            measurements.emplace_back(Json(std::move(row)));
        }

        Json::Object profile_row;
        profile_row["k"] = Json(static_cast<std::int64_t>(k));
        profile_row["perfect_draft_tokens"] = token_array(draft);
        profile_row["measurements"] = Json(std::move(measurements));
        profiles.emplace_back(Json(std::move(profile_row)));
    }

    Json::Object result = result_base(kProfileSchema, "profile", "ok");
    result["measurement_clock"] = Json("std::chrono::steady_clock/nanoseconds/v1");
    result["measurement_scope"] = Json("decode-and-synchronize/no-context-creation/v1");
    result["claim_boundary"] = Json("isolated-perfect-draft-verifier-profile/not-live-pcss/v1");
    result["seed_id"] = Json(seed.seed_id);
    result["warmup_repetitions"] = Json(static_cast<std::int64_t>(warmup_repetitions));
    result["measurement_repetitions"] = Json(static_cast<std::int64_t>(measurement_repetitions));
    result["prompt_state_sha256"] = Json(sha256_hex(prompt_state));
    result["prompt_state_bytes"] = Json(static_cast<std::int64_t>(prompt_state.size()));
    result["profiles"] = Json(std::move(profiles));
    result["configuration"] = runtime_echo(config);
    result["observed_context"] = observed_context(prompt);
    result["state_export"] = Json(kStateExport);
    result["state_api"] = Json(kStateApi);
    result["generation_mode"] = Json("greedy_argmax");
    result["greedy_tie_break"] = Json("lowest-token-id/v1");
    result["tokenization_boundary"] = Json("caller-supplied-raw-token-ids/v1");
    result["chat_template_applied"] = Json(false);
    result["thinking_mode_applied"] = Json(false);
    return {Json(std::move(result)), true};
}

HandledResponse profile_pending(const Json::Object & request) {
    require_exact_keys(
        request,
        {
            "schema_version",
            "operation",
            "model_artifact",
            "llama_cpp_library",
            "llama_cpp_revision",
            "runtime",
            "seed",
            "profile",
        }
    );
    require_request_schema(request, "profile_pending");
    const RequestConfig config = parse_config(request);
    const SeedSpec seed = parse_seed(request, false);
    const Json::Object & profile_config = as_object(required_field(request, "profile"), "profile");
    require_exact_keys(profile_config, {"k_values", "warmup_repetitions", "measurement_repetitions"});
    const std::vector<std::uint32_t> k_values = required_k_values(profile_config);
    const std::uint32_t warmup_repetitions = required_u32(profile_config, "warmup_repetitions", 0U);
    const std::uint32_t measurement_repetitions = required_u32(
        profile_config,
        "measurement_repetitions",
        2U
    );
    const std::uint32_t max_k = *std::max_element(k_values.begin(), k_values.end());
    if (static_cast<std::size_t>(max_k) + 1U > config.runtime.n_batch) {
        throw BridgeError("invalid_request", "pending profile anchor plus k exceeds configured n_batch");
    }
    if (seed.initial_context_tokens.size() > config.runtime.n_ctx ||
        static_cast<std::size_t>(max_k) + 1U >
            static_cast<std::size_t>(config.runtime.n_ctx) - seed.initial_context_tokens.size()) {
        throw BridgeError("invalid_request", "pending profile exceeds remaining n_ctx");
    }

    ModelRuntime runtime(config);
    runtime.validate_tokens(seed.initial_context_tokens, "seed.initial_context_tokens");
    const std::size_t prefix_size = seed.initial_context_tokens.size() - 1U;
    const llama_token anchor_token = seed.initial_context_tokens.back();
    const std::vector<llama_token> prefix_tokens(
        seed.initial_context_tokens.begin(),
        seed.initial_context_tokens.end() - 1
    );
    auto prefill_prefix = [&](Context & context) {
        if (!prefix_tokens.empty()) {
            decode_tokens(context, prefix_tokens, 0U, false);
        }
        llama_synchronize(context.raw());
    };

    Context prompt = runtime.make_context();
    prefill_prefix(prompt);
    const std::vector<std::uint8_t> prompt_state = snapshot_state(prompt);

    Context target = runtime.make_context();
    prefill_prefix(target);
    decode_tokens(target, std::vector<llama_token>{anchor_token}, prefix_size, true);
    const std::vector<llama_token> perfect_tokens = serial_greedy_tokens(
        target,
        prefix_size + 1U,
        static_cast<std::size_t>(max_k) + 1U,
        runtime.vocabulary_size()
    );

    Json::Array profiles;
    for (const std::uint32_t k : k_values) {
        const std::vector<llama_token> draft(perfect_tokens.begin(), perfect_tokens.begin() + k);
        std::vector<llama_token> verification_batch;
        verification_batch.reserve(draft.size() + 1U);
        verification_batch.push_back(anchor_token);
        verification_batch.insert(verification_batch.end(), draft.begin(), draft.end());

        Context serial_before = runtime.make_context();
        Context verifier = runtime.make_context();
        Context serial_after = runtime.make_context();
        Context restore_before_probe = runtime.make_context();
        Context restore_batch_probe = runtime.make_context();
        Context restore_after_probe = runtime.make_context();

        auto run_serial = [&](Context & context) {
            for (std::size_t index = 0; index < verification_batch.size(); ++index) {
                decode_tokens(
                    context,
                    std::vector<llama_token>{verification_batch[index]},
                    prefix_size + index,
                    true
                );
            }
            llama_synchronize(context.raw());
        };
        auto run_batch = [&](Context & context) {
            decode_one_batch_with_all_logits(context, verification_batch, prefix_size);
            llama_synchronize(context.raw());
        };
        auto require_exact_state = [&](Context & serial, Context & batch) {
            const std::vector<std::uint8_t> serial_state = snapshot_state(serial);
            const std::vector<std::uint8_t> batch_state = snapshot_state(batch);
            if (serial_state != batch_state ||
                greedy_argmax(serial, runtime.vocabulary_size()) != greedy_argmax(batch, runtime.vocabulary_size())) {
                throw BridgeError(
                    "profile_state_mismatch",
                    "pending-token batch state differs from the serial target state"
                );
            }
        };

        for (std::uint32_t iteration = 0; iteration < warmup_repetitions; ++iteration) {
            serial_before = runtime.make_context();
            prefill_prefix(serial_before);
            run_serial(serial_before);
            verifier = runtime.make_context();
            prefill_prefix(verifier);
            run_batch(verifier);
            require_exact_state(serial_before, verifier);
            serial_after = runtime.make_context();
            prefill_prefix(serial_after);
            run_serial(serial_after);
        }

        Json::Array measurements;
        for (std::uint32_t iteration = 0; iteration < measurement_repetitions; ++iteration) {
            serial_before = runtime.make_context();
            prefill_prefix(serial_before);
            const std::int64_t restore_before_ns = measured_nanoseconds([&]() {
                restore_state(restore_before_probe, prompt_state);
                llama_synchronize(restore_before_probe.raw());
            });
            const std::int64_t serial_before_ns = measured_nanoseconds([&]() { run_serial(serial_before); });
            verifier = runtime.make_context();
            prefill_prefix(verifier);
            const std::int64_t restore_batch_ns = measured_nanoseconds([&]() {
                restore_state(restore_batch_probe, prompt_state);
                llama_synchronize(restore_batch_probe.raw());
            });
            const std::int64_t batch_verify_ns = measured_nanoseconds([&]() { run_batch(verifier); });
            require_exact_state(serial_before, verifier);
            serial_after = runtime.make_context();
            prefill_prefix(serial_after);
            const std::int64_t restore_after_ns = measured_nanoseconds([&]() {
                restore_state(restore_after_probe, prompt_state);
                llama_synchronize(restore_after_probe.raw());
            });
            const std::int64_t serial_after_ns = measured_nanoseconds([&]() { run_serial(serial_after); });
            const std::int64_t snapshot_ns = measured_nanoseconds([&]() {
                const std::vector<std::uint8_t> state = snapshot_state(verifier);
                if (state.empty()) {
                    throw BridgeError("internal_error", "pending profile snapshot unexpectedly empty");
                }
            });

            Json::Object row;
            row["iteration"] = Json(static_cast<std::int64_t>(iteration));
            row["restore_before_ns"] = Json(restore_before_ns);
            row["serial_before_ns"] = Json(serial_before_ns);
            row["restore_batch_ns"] = Json(restore_batch_ns);
            row["batch_verify_ns"] = Json(batch_verify_ns);
            row["restore_after_ns"] = Json(restore_after_ns);
            row["serial_after_ns"] = Json(serial_after_ns);
            row["snapshot_ns"] = Json(snapshot_ns);
            measurements.emplace_back(Json(std::move(row)));
        }

        Json::Object profile_row;
        profile_row["k"] = Json(static_cast<std::int64_t>(k));
        profile_row["perfect_draft_tokens"] = token_array(draft);
        profile_row["measurements"] = Json(std::move(measurements));
        profiles.emplace_back(Json(std::move(profile_row)));
    }

    Json::Object result = result_base(kPendingProfileSchema, "profile_pending", "ok");
    result["measurement_clock"] = Json("std::chrono::steady_clock/nanoseconds/v1");
    result["measurement_scope"] = Json("pending-anchor-decode-and-synchronize/no-context-creation/v1");
    result["claim_boundary"] = Json("isolated-exact-state-pending-token-profile/not-live-pcss/v1");
    result["seed_id"] = Json(seed.seed_id);
    result["warmup_repetitions"] = Json(static_cast<std::int64_t>(warmup_repetitions));
    result["measurement_repetitions"] = Json(static_cast<std::int64_t>(measurement_repetitions));
    result["prompt_state_sha256"] = Json(sha256_hex(prompt_state));
    result["prompt_state_bytes"] = Json(static_cast<std::int64_t>(prompt_state.size()));
    result["profiles"] = Json(std::move(profiles));
    result["configuration"] = runtime_echo(config);
    result["observed_context"] = observed_context(prompt);
    result["state_export"] = Json(kStateExport);
    result["state_api"] = Json(kStateApi);
    result["generation_mode"] = Json("greedy_argmax");
    result["greedy_tie_break"] = Json("lowest-token-id/v1");
    result["tokenization_boundary"] = Json("caller-supplied-raw-token-ids/v1");
    result["chat_template_applied"] = Json(false);
    result["thinking_mode_applied"] = Json(false);
    return {Json(std::move(result)), true};
}

HandledResponse benchmark_pending(const Json::Object & request) {
    require_exact_keys(
        request,
        {
            "schema_version",
            "operation",
            "model_artifact",
            "llama_cpp_library",
            "llama_cpp_revision",
            "runtime",
            "seed",
            "benchmark",
        }
    );
    require_request_schema(request, "benchmark_pending");
    const RequestConfig config = parse_config(request);
    const SeedSpec seed = parse_seed(request, false);
    const Json::Object & benchmark = as_object(required_field(request, "benchmark"), "benchmark");
    require_exact_keys(
        benchmark,
        {
            "case_id", "candidate_tokens", "k", "output_tokens", "warmup_repetitions",
            "measurement_repetitions", "reanchor_min_tokens"
        }
    );
    const std::string case_id = required_string(benchmark, "case_id");
    const std::vector<llama_token> candidate_tokens = required_tokens(benchmark, "candidate_tokens");
    const std::size_t k = static_cast<std::size_t>(required_u32(benchmark, "k", 1U));
    const std::size_t output_count = static_cast<std::size_t>(required_u32(benchmark, "output_tokens", 2U));
    const std::size_t warmup_repetitions = static_cast<std::size_t>(
        required_u32(benchmark, "warmup_repetitions", 1U)
    );
    const std::size_t measurement_repetitions = static_cast<std::size_t>(
        required_u32(benchmark, "measurement_repetitions", 2U)
    );
    const std::size_t reanchor_min_tokens = static_cast<std::size_t>(
        required_u32(benchmark, "reanchor_min_tokens", 0U)
    );
    if (k + 1U > config.runtime.n_batch) {
        throw BridgeError("invalid_request", "pending benchmark batch exceeds configured n_batch");
    }
    const std::size_t prefix_size = seed.initial_context_tokens.size() - 1U;
    if (prefix_size + output_count > config.runtime.n_ctx) {
        throw BridgeError("invalid_request", "pending benchmark output exceeds configured n_ctx");
    }
    if (candidate_tokens.size() < output_count) {
        throw BridgeError("invalid_request", "pending benchmark candidate is shorter than requested output");
    }

    ModelRuntime runtime(config);
    runtime.validate_tokens(seed.initial_context_tokens, "seed.initial_context_tokens");
    runtime.validate_tokens(candidate_tokens, "benchmark.candidate_tokens");
    const llama_token anchor_token = seed.initial_context_tokens.back();
    const std::vector<llama_token> prefix_tokens(
        seed.initial_context_tokens.begin(),
        seed.initial_context_tokens.end() - 1
    );
    Context prompt = runtime.make_context();
    if (!prefix_tokens.empty()) {
        decode_tokens(prompt, prefix_tokens, 0U, false);
    }
    const std::vector<std::uint8_t> prompt_state = snapshot_state(prompt);

    struct DecodeRun {
        std::vector<llama_token> emitted;
        std::vector<std::uint8_t> state;
        std::int64_t decode_ns = 0;
        std::size_t proposed = 0U;
        std::size_t accepted = 0U;
        std::size_t verification_rounds = 0U;
        std::size_t serial_fallback_steps = 0U;
        std::size_t serial_state_reconstruction_rounds = 0U;
        std::size_t reanchor_attempts = 0U;
        std::size_t reanchor_successes = 0U;
    };
    auto run_serial = [&](Context & context) {
        DecodeRun run;
        run.emitted.reserve(output_count);
        llama_token pending = anchor_token;
        run.decode_ns = measured_nanoseconds([&]() {
            for (std::size_t index = 0; index < output_count; ++index) {
                decode_tokens(context, std::vector<llama_token>{pending}, prefix_size + index, true);
                pending = greedy_argmax(context, runtime.vocabulary_size());
                run.emitted.push_back(pending);
            }
        });
        run.state = snapshot_state(context);
        return run;
    };
    auto run_candidate = [&](Context & context) {
        DecodeRun run;
        run.emitted.reserve(output_count);
        llama_token pending = anchor_token;
        std::size_t committed_size = prefix_size;
        std::size_t source_cursor = 0U;
        bool candidate_enabled = true;
        auto try_reanchor = [&]() {
            if (candidate_enabled || reanchor_min_tokens == 0U || run.emitted.size() < reanchor_min_tokens) {
                return;
            }
            ++run.reanchor_attempts;
            const auto suffix_begin = run.emitted.end() - static_cast<std::ptrdiff_t>(reanchor_min_tokens);
            for (std::size_t position = source_cursor;
                 position + reanchor_min_tokens <= candidate_tokens.size();
                 ++position) {
                if (std::equal(
                        suffix_begin,
                        run.emitted.end(),
                        candidate_tokens.begin() + static_cast<std::ptrdiff_t>(position)
                    )) {
                    source_cursor = position + reanchor_min_tokens;
                    candidate_enabled = source_cursor < candidate_tokens.size();
                    if (candidate_enabled) {
                        ++run.reanchor_successes;
                    }
                    return;
                }
            }
        };
        run.decode_ns = measured_nanoseconds([&]() {
            while (run.emitted.size() < output_count) {
                const std::size_t remaining = output_count - run.emitted.size();
                const std::size_t candidate_remaining = candidate_tokens.size() - source_cursor;
                const std::size_t draft_count = candidate_enabled && remaining > 1U
                    ? std::min({k, remaining - 1U, candidate_remaining})
                    : 0U;
                if (draft_count == 0U) {
                    decode_tokens(context, std::vector<llama_token>{pending}, committed_size, true);
                    pending = greedy_argmax(context, runtime.vocabulary_size());
                    run.emitted.push_back(pending);
                    ++committed_size;
                    ++run.serial_fallback_steps;
                    try_reanchor();
                    continue;
                }
                std::vector<llama_token> batch;
                batch.reserve(draft_count + 1U);
                batch.push_back(pending);
                batch.insert(
                    batch.end(),
                    candidate_tokens.begin() + static_cast<std::ptrdiff_t>(source_cursor),
                    candidate_tokens.begin() + static_cast<std::ptrdiff_t>(source_cursor + draft_count)
                );
                const std::vector<std::uint8_t> pre_verification_state = snapshot_state(context);
                decode_one_batch_with_all_logits(context, batch, committed_size);
                std::vector<llama_token> targets;
                targets.reserve(batch.size());
                for (std::size_t index = 0; index < batch.size(); ++index) {
                    targets.push_back(
                        greedy_argmax_at(context, runtime.vocabulary_size(), static_cast<std::int32_t>(index))
                    );
                }
                std::size_t accepted = 0U;
                while (accepted < draft_count && batch[accepted + 1U] == targets[accepted]) {
                    ++accepted;
                }
                run.proposed += draft_count;
                run.accepted += accepted;
                ++run.verification_rounds;
                llama_synchronize(context.raw());
                if (accepted > 2U) {
                    restore_state(context, pre_verification_state);
                    decode_tokens(context, std::vector<llama_token>{pending}, committed_size, true);
                    for (std::size_t index = 0; index < accepted; ++index) {
                        decode_tokens(
                            context,
                            std::vector<llama_token>{batch[index + 1U]},
                            committed_size + 1U + index,
                            true
                        );
                    }
                    ++run.serial_state_reconstruction_rounds;
                } else {
                    const llama_pos trim_from = static_cast<llama_pos>(committed_size + 1U + accepted);
                    if (!llama_memory_seq_rm(llama_get_memory(context.raw()), 0, trim_from, -1)) {
                        throw BridgeError("state_trim_unsupported", "llama.cpp cannot remove a live benchmark tail");
                    }
                }
                for (std::size_t index = 0; index < accepted; ++index) {
                    run.emitted.push_back(batch[index + 1U]);
                }
                source_cursor += accepted;
                committed_size += 1U + accepted;
                if (accepted < draft_count) {
                    pending = targets[accepted];
                    run.emitted.push_back(pending);
                    candidate_enabled = false;
                    try_reanchor();
                } else {
                    pending = targets[draft_count];
                    run.emitted.push_back(pending);
                    if (source_cursor < candidate_tokens.size() && candidate_tokens[source_cursor] == pending) {
                        ++source_cursor;
                    } else {
                        candidate_enabled = false;
                        try_reanchor();
                    }
                }
            }
        });
        run.state = snapshot_state(context);
        return run;
    };
    auto require_equal = [&](const DecodeRun & serial, const DecodeRun & candidate, const char * phase) {
        if (serial.emitted != candidate.emitted) {
            throw BridgeError("live_output_mismatch", std::string(phase) + " output differs from serial greedy");
        }
        if (serial.state != candidate.state) {
            throw BridgeError("live_state_mismatch", std::string(phase) + " state differs from serial greedy");
        }
    };

    Context serial_before = runtime.make_context();
    Context candidate = runtime.make_context();
    Context serial_after = runtime.make_context();
    for (std::size_t index = 0; index < warmup_repetitions; ++index) {
        restore_state(serial_before, prompt_state);
        restore_state(candidate, prompt_state);
        const DecodeRun serial = run_serial(serial_before);
        const DecodeRun speculative = run_candidate(candidate);
        require_equal(serial, speculative, "warmup");
    }
    Json::Array measurements;
    measurements.reserve(measurement_repetitions);
    std::vector<llama_token> authoritative_tokens;
    std::vector<std::uint8_t> authoritative_state;
    for (std::size_t index = 0; index < measurement_repetitions; ++index) {
        restore_state(serial_before, prompt_state);
        restore_state(candidate, prompt_state);
        restore_state(serial_after, prompt_state);
        const DecodeRun before = run_serial(serial_before);
        const DecodeRun speculative = run_candidate(candidate);
        const DecodeRun after = run_serial(serial_after);
        require_equal(before, speculative, "measurement");
        require_equal(after, speculative, "measurement repeat");
        if (before.emitted != after.emitted || before.state != after.state) {
            throw BridgeError("baseline_repeat_mismatch", "paired serial live benchmark baselines differ");
        }
        authoritative_tokens = before.emitted;
        authoritative_state = before.state;
        Json::Object row;
        row["iteration"] = Json(static_cast<std::int64_t>(index));
        row["serial_before_ns"] = Json(before.decode_ns);
        row["candidate_decode_ns"] = Json(speculative.decode_ns);
        row["serial_after_ns"] = Json(after.decode_ns);
        row["proposed_tokens"] = Json(static_cast<std::int64_t>(speculative.proposed));
        row["accepted_tokens"] = Json(static_cast<std::int64_t>(speculative.accepted));
        row["verification_rounds"] = Json(static_cast<std::int64_t>(speculative.verification_rounds));
        row["serial_fallback_steps"] = Json(static_cast<std::int64_t>(speculative.serial_fallback_steps));
        row["serial_state_reconstruction_rounds"] = Json(
            static_cast<std::int64_t>(speculative.serial_state_reconstruction_rounds)
        );
        row["reanchor_attempts"] = Json(static_cast<std::int64_t>(speculative.reanchor_attempts));
        row["reanchor_successes"] = Json(static_cast<std::int64_t>(speculative.reanchor_successes));
        measurements.emplace_back(Json(std::move(row)));
    }

    Json::Object result = result_base(kPendingBenchmarkSchema, "benchmark_pending", "ok");
    result["case_id"] = Json(case_id);
    result["seed_id"] = Json(seed.seed_id);
    result["k"] = Json(static_cast<std::int64_t>(k));
    result["output_tokens"] = Json(static_cast<std::int64_t>(output_count));
    result["warmup_repetitions"] = Json(static_cast<std::int64_t>(warmup_repetitions));
    result["measurement_repetitions"] = Json(static_cast<std::int64_t>(measurement_repetitions));
    result["reanchor_min_tokens"] = Json(static_cast<std::int64_t>(reanchor_min_tokens));
    result["authoritative_tokens"] = token_array(authoritative_tokens);
    result["candidate_prefix_tokens"] = token_array(
        std::vector<llama_token>(candidate_tokens.begin(), candidate_tokens.begin() + output_count)
    );
    result["final_state_sha256"] = Json(sha256_hex(authoritative_state));
    result["final_state_bytes"] = Json(static_cast<std::int64_t>(authoritative_state.size()));
    result["measurements"] = Json(std::move(measurements));
    result["measurement_clock"] = Json("std::chrono::steady_clock/nanoseconds/v1");
    result["measurement_scope"] = Json("decode-and-synchronize/prefill-restore-provider-excluded/v1");
    result["provider_integration"] = Json("static-trace-candidate/provider-lookup-excluded/v1");
    result["exact_output_and_state"] = Json(true);
    result["configuration"] = runtime_echo(config);
    result["observed_context"] = observed_context(candidate);
    result["state_export"] = Json(kStateExport);
    result["state_api"] = Json(kStateApi);
    result["generation_mode"] = Json("greedy_argmax");
    result["greedy_tie_break"] = Json("lowest-token-id/v1");
    result["tokenization_boundary"] = Json("caller-supplied-raw-token-ids/v1");
    result["chat_template_applied"] = Json(false);
    result["thinking_mode_applied"] = Json(false);
    result["claim_boundary"] = Json("controlled-static-candidate-live-decode/not-general-pcss/v1");
    return {Json(std::move(result)), true};
}

HandledResponse handle_request(const Json & request, const char * expected_operation) {
    const Json::Object & object = as_object(request, "request");
    const std::string operation = required_string(object, "operation");
    if (operation != expected_operation) {
        throw BridgeError("invalid_request", "argv operation guard does not match request.operation");
    }
    if (operation == "capabilities") {
        return capabilities(object);
    }
    if (operation == "reference") {
        return reference(object);
    }
    if (operation == "reference_pending") {
        return reference(object, true);
    }
    if (operation == "verify") {
        return verify(object);
    }
    if (operation == "verify_pending") {
        return verify_pending(object);
    }
    if (operation == "trajectory_pending") {
        return trajectory_pending(object);
    }
    if (operation == "profile") {
        return profile(object);
    }
    if (operation == "profile_pending") {
        return profile_pending(object);
    }
    if (operation == "benchmark_pending") {
        return benchmark_pending(object);
    }
    throw BridgeError("invalid_request", "unsupported bridge operation");
}

Json error_response(const char * operation, const BridgeError & error) {
    Json::Object error_payload;
    error_payload["code"] = Json(error.code());
    error_payload["message"] = Json(error.what());
    Json::Object result = result_base(kResultSchema, operation, "error");
    result["error"] = Json(std::move(error_payload));
    return Json(std::move(result));
}

Json internal_error_response(const char * operation) {
    Json::Object error_payload;
    error_payload["code"] = Json("internal_error");
    error_payload["message"] = Json("bridge encountered an unexpected internal error");
    Json::Object result = result_base(kResultSchema, operation, "error");
    result["error"] = Json(std::move(error_payload));
    return Json(std::move(result));
}

const char * operation_from_argv(int argc, char ** argv) {
    if (argc != 2) {
        throw BridgeError(
            "invalid_invocation",
            "use exactly one operation guard: --pcss-capabilities, --pcss-reference, --pcss-reference-pending, --pcss-verify-case, --pcss-verify-pending-case, --pcss-trajectory-pending, --pcss-profile, --pcss-profile-pending, or --pcss-benchmark-pending"
        );
    }
    const std::string argument(argv[1]);
    if (argument == "--pcss-capabilities") {
        return "capabilities";
    }
    if (argument == "--pcss-reference") {
        return "reference";
    }
    if (argument == "--pcss-reference-pending") {
        return "reference_pending";
    }
    if (argument == "--pcss-verify-case") {
        return "verify";
    }
    if (argument == "--pcss-verify-pending-case") {
        return "verify_pending";
    }
    if (argument == "--pcss-trajectory-pending") {
        return "trajectory_pending";
    }
    if (argument == "--pcss-profile") {
        return "profile";
    }
    if (argument == "--pcss-profile-pending") {
        return "profile_pending";
    }
    if (argument == "--pcss-benchmark-pending") {
        return "benchmark_pending";
    }
    throw BridgeError(
        "invalid_invocation",
        "use exactly one operation guard: --pcss-capabilities, --pcss-reference, --pcss-reference-pending, --pcss-verify-case, --pcss-verify-pending-case, --pcss-trajectory-pending, --pcss-profile, --pcss-profile-pending, or --pcss-benchmark-pending"
    );
}

}  // namespace

int main(int argc, char ** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    const char * expected_operation = "unknown";
    try {
        expected_operation = operation_from_argv(argc, argv);
    } catch (const BridgeError & error) {
        std::cout << json_dump(error_response(expected_operation, error)) << '\n';
        return 64;
    }

    llama_backend_init();
    bool failed = false;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        if (line.size() > kMaxRequestLineBytes) {
            std::cout << json_dump(error_response(expected_operation, BridgeError(
                "invalid_request",
                "request line exceeds the bridge limit"
            ))) << '\n';
            failed = true;
            continue;
        }
        try {
            const Json request = JsonParser(line).parse_document();
            const HandledResponse response = handle_request(request, expected_operation);
            std::cout << json_dump(response.payload) << '\n';
            if (!response.successful) {
                failed = true;
            }
        } catch (const BridgeError & error) {
            std::cout << json_dump(error_response(expected_operation, error)) << '\n';
            failed = true;
        } catch (const std::exception &) {
            std::cout << json_dump(internal_error_response(expected_operation)) << '\n';
            failed = true;
        }
        std::cout.flush();
    }
    llama_backend_free();
    return failed ? 1 : 0;
}
