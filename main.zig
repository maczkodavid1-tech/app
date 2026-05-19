const std = @import("std");
const llm = @import("llm.zig");
const exa = @import("exa.zig");

pub const MAX_SESSIONS: usize = 500;
pub const SESSION_TTL: f64 = 21600.0;
pub const MAX_TOOL_ITERATIONS: usize = 20;
pub const MAX_HISTORY_MESSAGES: usize = 60;
pub const MAX_STORED_MESSAGES: usize = 120;
pub const MAX_SUMMARY_CHARS: usize = 600;
pub const EXA_NUM_RESULTS: usize = 100;
pub const MAX_EXA_CONCURRENT: usize = 5;
pub const MAX_MODEL_CONTENT_CHARS: usize = 180000;
pub const MAX_DELETED_SESSIONS: usize = 1000;
pub const MODEL_CONTEXT_LIMIT: usize = 202752;
pub const DESIRED_MAX_TOKENS: usize = 180000;
pub const MAX_MESSAGE_LENGTH: usize = 100000;
pub const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
pub const EXA_QPS_LIMIT: usize = 8;
pub const SESSION_PING_INTERVAL_MS: u64 = 5000;
const AUTH_TOKEN_TTL: f64 = 30.0 * 24.0 * 3600.0;
const AUTH_CHALLENGE_TTL: f64 = 300.0;

pub const ExaToolResult = struct {
    events: std.ArrayList([]u8),
    tool_msg: Message,
    tool_type: []const u8,
};

pub const IMAGE_REF_DELIM = "|||";

pub const ALLOWED_IMAGE_MIMES = [_][]const u8{
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
};

pub const DEEP_SEARCH_TYPES = [_][]const u8{ "deep", "deep-reasoning" };

pub const MimeExt = struct {
    mime: []const u8,
    ext: []const u8,
};

pub const MIME_TO_EXT = [_]MimeExt{
    .{ .mime = "image/jpeg", .ext = ".jpg" },
    .{ .mime = "image/png", .ext = ".png" },
    .{ .mime = "image/webp", .ext = ".webp" },
    .{ .mime = "image/gif", .ext = ".gif" },
};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
pub var global_allocator: std.mem.Allocator = undefined;

pub var hpc_ai_api_key: []const u8 = "";
pub var exa_api_key: []const u8 = "";
pub var project_root: []const u8 = "";
var index_file: []const u8 = "";
var static_dir: []const u8 = "";
var static_sw_file: []const u8 = "";
var persistence_file: []const u8 = "";
var auth_file: []const u8 = "";
var server_host: []const u8 = "";
var server_port: u16 = 5000;

pub const MessageContentPart = struct {
    type: []u8,
    text: ?[]u8,
    media_type: ?[]u8,
    data: ?[]u8,
    key: ?[]u8,
    image_url: ?ImageUrl,

    pub const ImageUrl = struct {
        url: []u8,
        detail: []u8,
    };

    pub fn deinit(self: *MessageContentPart, allocator: std.mem.Allocator) void {
        allocator.free(self.type);
        if (self.text) |t| allocator.free(t);
        if (self.media_type) |m| allocator.free(m);
        if (self.data) |d| allocator.free(d);
        if (self.key) |k| allocator.free(k);
        if (self.image_url) |*iu| {
            allocator.free(iu.url);
            allocator.free(iu.detail);
        }
    }

    pub fn clone(self: MessageContentPart, allocator: std.mem.Allocator) !MessageContentPart {
        const type_copy = try allocator.dupe(u8, self.type);
        errdefer allocator.free(type_copy);

        const text_copy: ?[]u8 = if (self.text) |t| try allocator.dupe(u8, t) else null;
        errdefer if (text_copy) |t| allocator.free(t);

        const media_type_copy: ?[]u8 = if (self.media_type) |m| try allocator.dupe(u8, m) else null;
        errdefer if (media_type_copy) |m| allocator.free(m);

        const data_copy: ?[]u8 = if (self.data) |d| try allocator.dupe(u8, d) else null;
        errdefer if (data_copy) |d| allocator.free(d);

        const key_copy: ?[]u8 = if (self.key) |k| try allocator.dupe(u8, k) else null;
        errdefer if (key_copy) |k| allocator.free(k);

        const image_url_copy: ?ImageUrl = if (self.image_url) |iu| blk: {
            const url = try allocator.dupe(u8, iu.url);
            errdefer allocator.free(url);
            const detail = try allocator.dupe(u8, iu.detail);
            break :blk ImageUrl{ .url = url, .detail = detail };
        } else null;

        return MessageContentPart{
            .type = type_copy,
            .text = text_copy,
            .media_type = media_type_copy,
            .data = data_copy,
            .key = key_copy,
            .image_url = image_url_copy,
        };
    }
};

pub const ToolCallFunction = struct {
    name: []u8,
    arguments: []u8,

    pub fn deinit(self: *ToolCallFunction, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.arguments);
    }

    pub fn clone(self: ToolCallFunction, allocator: std.mem.Allocator) !ToolCallFunction {
        const name = try allocator.dupe(u8, self.name);
        errdefer allocator.free(name);
        const arguments = try allocator.dupe(u8, self.arguments);
        return ToolCallFunction{ .name = name, .arguments = arguments };
    }
};

pub const ToolCall = struct {
    id: []u8,
    type: []u8,
    function: ToolCallFunction,

    pub fn deinit(self: *ToolCall, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.type);
        self.function.deinit(allocator);
    }

    pub fn clone(self: ToolCall, allocator: std.mem.Allocator) !ToolCall {
        const id = try allocator.dupe(u8, self.id);
        errdefer allocator.free(id);
        const type_copy = try allocator.dupe(u8, self.type);
        errdefer allocator.free(type_copy);
        const function = try self.function.clone(allocator);
        return ToolCall{ .id = id, .type = type_copy, .function = function };
    }
};

pub const MessageContent = union(enum) {
    text: []u8,
    parts: std.ArrayList(MessageContentPart),

    pub fn deinit(self: *MessageContent, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .text => |t| allocator.free(t),
            .parts => |*p| {
                for (p.items) |*part| part.deinit(allocator);
                p.deinit();
            },
        }
    }

    pub fn clone(self: MessageContent, allocator: std.mem.Allocator) !MessageContent {
        switch (self) {
            .text => |t| return MessageContent{ .text = try allocator.dupe(u8, t) },
            .parts => |p| {
                var new_parts = std.ArrayList(MessageContentPart).init(allocator);
                errdefer {
                    for (new_parts.items) |*part| part.deinit(allocator);
                    new_parts.deinit();
                }
                for (p.items) |part| try new_parts.append(try part.clone(allocator));
                return MessageContent{ .parts = new_parts };
            },
        }
    }
};

pub const Message = struct {
    role: []u8,
    content: MessageContent,
    tool_calls: ?std.ArrayList(ToolCall),
    tool_call_id: ?[]u8,
    msg_id: ?[]u8,
    cached_size: ?usize,

    pub fn deinit(self: *Message, allocator: std.mem.Allocator) void {
        allocator.free(self.role);
        self.content.deinit(allocator);
        if (self.tool_calls) |*tcs| {
            for (tcs.items) |*tc| tc.deinit(allocator);
            tcs.deinit();
        }
        if (self.tool_call_id) |id| allocator.free(id);
        if (self.msg_id) |mid| allocator.free(mid);
    }

    pub fn clone(self: Message, allocator: std.mem.Allocator) !Message {
        const role = try allocator.dupe(u8, self.role);
        errdefer allocator.free(role);

        var content = try self.content.clone(allocator);
        errdefer content.deinit(allocator);

        const tool_call_id: ?[]u8 = if (self.tool_call_id) |id| try allocator.dupe(u8, id) else null;
        errdefer if (tool_call_id) |id| allocator.free(id);

        const msg_id: ?[]u8 = if (self.msg_id) |mid| try allocator.dupe(u8, mid) else null;
        errdefer if (msg_id) |mid| allocator.free(mid);

        var tool_calls: ?std.ArrayList(ToolCall) = null;
        if (self.tool_calls) |tcs| {
            var new_tcs = std.ArrayList(ToolCall).init(allocator);
            errdefer {
                for (new_tcs.items) |*tc| tc.deinit(allocator);
                new_tcs.deinit();
            }
            for (tcs.items) |tc| try new_tcs.append(try tc.clone(allocator));
            tool_calls = new_tcs;
        }

        return Message{
            .role = role,
            .content = content,
            .tool_calls = tool_calls,
            .tool_call_id = tool_call_id,
            .msg_id = msg_id,
            .cached_size = self.cached_size,
        };
    }
};

pub const Session = struct {
    messages: std.ArrayList(Message),
    created_at: f64,
    updated_at: f64,

    pub fn init(allocator: std.mem.Allocator, created_at: f64) Session {
        return Session{
            .messages = std.ArrayList(Message).init(allocator),
            .created_at = created_at,
            .updated_at = created_at,
        };
    }

    pub fn deinit(self: *Session, allocator: std.mem.Allocator) void {
        for (self.messages.items) |*msg| msg.deinit(allocator);
        self.messages.deinit();
    }
};

pub const RateLimiter = struct {
    max: usize,
    timestamps: std.ArrayList(i64),
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator, max_per_second: usize) RateLimiter {
        return RateLimiter{
            .max = max_per_second,
            .timestamps = std.ArrayList(i64).init(allocator),
            .mutex = .{},
        };
    }

    pub fn deinit(self: *RateLimiter) void {
        self.mutex.lock();
        self.timestamps.deinit();
        self.mutex.unlock();
    }

    pub fn acquire(self: *RateLimiter) void {
        while (true) {
            const now_ns = std.time.nanoTimestamp();
            const now_ms: i64 = @intCast(@divTrunc(now_ns, 1_000_000));
            const window_ms: i64 = 1000;
            const wait_ms: i64 = blk: {
                self.mutex.lock();
                defer self.mutex.unlock();
                var i: usize = 0;
                while (i < self.timestamps.items.len) {
                    if (now_ms - self.timestamps.items[i] >= window_ms) {
                        _ = self.timestamps.swapRemove(i);
                    } else {
                        i += 1;
                    }
                }
                if (self.timestamps.items.len < self.max) {
                    self.timestamps.append(now_ms) catch {
                        break :blk 0;
                    };
                    return;
                }
                var oldest: i64 = now_ms;
                for (self.timestamps.items) |ts| if (ts < oldest) { oldest = ts; };
                const wait_until = oldest + window_ms + 50;
                break :blk @max(0, wait_until - now_ms);
            };
            if (wait_ms <= 0) continue;
            std.time.sleep(@intCast(wait_ms * 1_000_000));
        }
    }
};

pub var sessions_map: std.StringHashMap(Session) = undefined;
pub var sessions_mutex: std.Thread.Mutex = .{};
var sessions_disk_mutex: std.Thread.Mutex = .{};

var session_locks_map: std.StringHashMap(*std.Thread.Mutex) = undefined;
var session_locks_guard: std.Thread.Mutex = .{};

var deleted_sessions_map: std.StringHashMap(f64) = undefined;
var deleted_sessions_mutex: std.Thread.Mutex = .{};

pub var exa_limiter: RateLimiter = undefined;
var session_exa_limiters_map: std.StringHashMap(*RateLimiter) = undefined;
var session_exa_limiters_mutex: std.Thread.Mutex = .{};

const AuthPasskey = struct {
    credential_id: []u8,
    public_key: []u8,
    sign_count: u32,
    rp_id: []u8,

    fn deinit(self: *AuthPasskey, allocator: std.mem.Allocator) void {
        allocator.free(self.credential_id);
        allocator.free(self.public_key);
        allocator.free(self.rp_id);
    }

    fn clone(self: AuthPasskey, allocator: std.mem.Allocator) !AuthPasskey {
        const cid = try allocator.dupe(u8, self.credential_id);
        errdefer allocator.free(cid);
        const pk = try allocator.dupe(u8, self.public_key);
        errdefer allocator.free(pk);
        const rp = try allocator.dupe(u8, self.rp_id);
        return AuthPasskey{
            .credential_id = cid,
            .public_key = pk,
            .sign_count = self.sign_count,
            .rp_id = rp,
        };
    }
};

const AuthUser = struct {
    id: []u8,
    email: []u8,
    password_hash: []u8,
    passkeys: std.ArrayList(AuthPasskey),
    created_at: f64,

    fn deinit(self: *AuthUser, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.email);
        allocator.free(self.password_hash);
        for (self.passkeys.items) |*pk| pk.deinit(allocator);
        self.passkeys.deinit();
    }
};

const AuthTokenInfo = struct {
    email: []u8,
    expires_at: f64,

    fn deinit(self: *AuthTokenInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.email);
    }
};

const AuthChallengeInfo = struct {
    email: ?[]u8,
    is_login: bool,
    created_at: f64,
    origin: []u8,

    fn deinit(self: *AuthChallengeInfo, allocator: std.mem.Allocator) void {
        if (self.email) |e| allocator.free(e);
        allocator.free(self.origin);
    }
};

var auth_users: std.StringHashMap(AuthUser) = undefined;
var auth_tokens: std.StringHashMap(AuthTokenInfo) = undefined;
var auth_challenges: std.StringHashMap(AuthChallengeInfo) = undefined;
var auth_mutex: std.Thread.Mutex = .{};
var auth_disk_mutex: std.Thread.Mutex = .{};

var cred_id_to_email: std.StringHashMap([]u8) = undefined;
var cred_id_map_mutex: std.Thread.Mutex = .{};

pub fn getEnvAlloc(allocator: std.mem.Allocator, key: []const u8, default: []const u8) ![]u8 {
    const val = std.process.getEnvVarOwned(allocator, key) catch |err| {
        if (err == error.EnvironmentVariableNotFound) return allocator.dupe(u8, default);
        return err;
    };
    return val;
}

pub fn nowSeconds() f64 {
    return @as(f64, @floatFromInt(std.time.milliTimestamp())) / 1000.0;
}

pub fn generateUuid(allocator: std.mem.Allocator) ![]u8 {
    var bytes: [16]u8 = undefined;
    std.crypto.random.bytes(&bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    return std.fmt.allocPrint(allocator, "{x:0>2}{x:0>2}{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}", .{
        bytes[0],  bytes[1],  bytes[2],  bytes[3],
        bytes[4],  bytes[5],
        bytes[6],  bytes[7],
        bytes[8],  bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    });
}

pub fn generateShortHex(allocator: std.mem.Allocator) ![]u8 {
    var bytes: [4]u8 = undefined;
    std.crypto.random.bytes(&bytes);
    return std.fmt.allocPrint(allocator, "{x:0>2}{x:0>2}{x:0>2}{x:0>2}", .{ bytes[0], bytes[1], bytes[2], bytes[3] });
}

pub fn detectImageFormat(data: []const u8) struct { mime: []const u8, ext: []const u8 } {
    if (data.len >= 8 and std.mem.eql(u8, data[0..8], "\x89PNG\r\n\x1a\n")) return .{ .mime = "image/png", .ext = ".png" };
    if (data.len >= 2 and data[0] == 0xff and data[1] == 0xd8) return .{ .mime = "image/jpeg", .ext = ".jpg" };
    if (data.len >= 12 and std.mem.eql(u8, data[0..4], "RIFF") and std.mem.eql(u8, data[8..12], "WEBP")) return .{ .mime = "image/webp", .ext = ".webp" };
    if (data.len >= 6 and (std.mem.eql(u8, data[0..6], "GIF87a") or std.mem.eql(u8, data[0..6], "GIF89a"))) return .{ .mime = "image/gif", .ext = ".gif" };
    return .{ .mime = "application/octet-stream", .ext = "" };
}

pub fn isMimeAllowed(mime: []const u8) bool {
    for (ALLOWED_IMAGE_MIMES) |allowed| if (std.mem.eql(u8, mime, allowed)) return true;
    return false;
}

pub fn isDeepSearchType(t: []const u8) bool {
    for (DEEP_SEARCH_TYPES) |ds| if (std.mem.eql(u8, t, ds)) return true;
    return false;
}

pub fn mimeToExt(mime: []const u8) []const u8 {
    for (MIME_TO_EXT) |me| if (std.mem.eql(u8, me.mime, mime)) return me.ext;
    return ".img";
}

pub fn extToMime(ext: []const u8) ?[]const u8 {
    for (MIME_TO_EXT) |me| if (std.mem.eql(u8, me.ext, ext)) return me.mime;
    return null;
}

fn stripWhitespace(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var result = try std.ArrayList(u8).initCapacity(allocator, s.len);
    defer result.deinit();
    for (s) |c| if (c != ' ' and c != '\t' and c != '\n' and c != '\r') try result.append(c);
    return result.toOwnedSlice();
}

fn safeB64Decode(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    const stripped = try stripWhitespace(allocator, s);
    defer allocator.free(stripped);

    if (stripped.len == 0) return error.EmptyBase64AfterStrip;

    const pad = 4 - (stripped.len % 4);
    var padded: []u8 = stripped;
    var padded_owned = false;
    if (pad != 4) {
        padded = try allocator.alloc(u8, stripped.len + pad);
        padded_owned = true;
        @memcpy(padded[0..stripped.len], stripped);
        for (stripped.len..padded.len) |i| padded[i] = '=';
    }
    defer if (padded_owned) allocator.free(padded);

    const is_url_safe = std.mem.indexOfAny(u8, padded, "-_") != null;

    if (is_url_safe) {
        const converted = try allocator.dupe(u8, padded);
        defer allocator.free(converted);
        for (converted) |*c| {
            if (c.* == '-') c.* = '+';
            if (c.* == '_') c.* = '/';
        }
        const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(converted) catch return error.InvalidBase64;
        const decoded = try allocator.alloc(u8, decoded_size);
        std.base64.standard.Decoder.decode(decoded, converted) catch {
            allocator.free(decoded);
            return error.InvalidBase64;
        };
        return decoded;
    } else {
        const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(padded) catch return error.InvalidBase64;
        const decoded = try allocator.alloc(u8, decoded_size);
        std.base64.standard.Decoder.decode(decoded, padded) catch {
            allocator.free(decoded);
            return error.InvalidBase64;
        };
        return decoded;
    }
}

fn decodeAndStripB64(allocator: std.mem.Allocator, raw: []const u8) !struct { stripped: []u8, decoded: []u8 } {
    if (raw.len == 0) return error.EmptyBase64;
    var working: []const u8 = raw;
    if (std.mem.indexOf(u8, working, "data:")) |_| {
        if (std.mem.indexOf(u8, working, ",")) |comma_idx| working = working[comma_idx + 1 ..];
    }
    const stripped = try stripWhitespace(allocator, working);
    if (stripped.len == 0) { allocator.free(stripped); return error.EmptyBase64AfterStrip; }
    const decoded = safeB64Decode(allocator, stripped) catch { allocator.free(stripped); return error.InvalidBase64Encoding; };
    return .{ .stripped = stripped, .decoded = decoded };
}

pub fn pathInProject(full_path: []const u8) bool {
    if (project_root.len == 0) return false;
    if (std.mem.eql(u8, full_path, project_root)) return true;
    if (!std.mem.startsWith(u8, full_path, project_root)) return false;
    const sep = std.fs.path.sep;
    if (project_root[project_root.len - 1] == sep) return true;
    if (full_path.len <= project_root.len) return false;
    return full_path[project_root.len] == sep;
}

fn pruneDeletedSessionsUnlocked(now: f64) void {
    var to_remove = std.ArrayList([]const u8).init(global_allocator);
    defer to_remove.deinit();
    var it = deleted_sessions_map.iterator();
    while (it.next()) |entry| if (now - entry.value_ptr.* > SESSION_TTL) to_remove.append(entry.key_ptr.*) catch {};
    for (to_remove.items) |key| if (deleted_sessions_map.fetchRemove(key)) |kv| global_allocator.free(kv.key);
    if (deleted_sessions_map.count() > MAX_DELETED_SESSIONS) {
        var oldest_key: ?[]const u8 = null;
        var oldest_val: f64 = std.math.floatMax(f64);
        var it2 = deleted_sessions_map.iterator();
        while (it2.next()) |entry| {
            if (entry.value_ptr.* < oldest_val) {
                oldest_val = entry.value_ptr.*;
                oldest_key = entry.key_ptr.*;
            }
        }
        if (oldest_key) |k| if (deleted_sessions_map.fetchRemove(k)) |kv| global_allocator.free(kv.key);
    }
}

pub fn isDeleted(sid: []const u8) bool {
    deleted_sessions_mutex.lock();
    defer deleted_sessions_mutex.unlock();
    pruneDeletedSessionsUnlocked(nowSeconds());
    const deleted_at = deleted_sessions_map.get(sid) orelse return false;
    if (nowSeconds() - deleted_at > SESSION_TTL) return false;
    return true;
}

fn markDeleted(sid: []const u8) void {
    deleted_sessions_mutex.lock();
    defer deleted_sessions_mutex.unlock();
    pruneDeletedSessionsUnlocked(nowSeconds());
    if (deleted_sessions_map.contains(sid)) {
        deleted_sessions_map.put(sid, nowSeconds()) catch {};
        return;
    }
    const key = global_allocator.dupe(u8, sid) catch return;
    deleted_sessions_map.put(key, nowSeconds()) catch { global_allocator.free(key); };
}

fn unmarkDeleted(sid: []const u8) void {
    deleted_sessions_mutex.lock();
    defer deleted_sessions_mutex.unlock();
    if (deleted_sessions_map.fetchRemove(sid)) |kv| global_allocator.free(kv.key);
    pruneDeletedSessionsUnlocked(nowSeconds());
}

pub fn getOrCreateSessionLock(sid: []const u8) !*std.Thread.Mutex {
    session_locks_guard.lock();
    defer session_locks_guard.unlock();
    if (session_locks_map.get(sid)) |lock| return lock;
    const lock = try global_allocator.create(std.Thread.Mutex);
    errdefer global_allocator.destroy(lock);
    lock.* = .{};
    const key = try global_allocator.dupe(u8, sid);
    errdefer global_allocator.free(key);
    try session_locks_map.put(key, lock);
    return lock;
}

pub fn getOrCreateExaLimiter(sid: []const u8) !*RateLimiter {
    session_exa_limiters_mutex.lock();
    defer session_exa_limiters_mutex.unlock();
    if (session_exa_limiters_map.get(sid)) |limiter| return limiter;
    const limiter = try global_allocator.create(RateLimiter);
    errdefer global_allocator.destroy(limiter);
    limiter.* = RateLimiter.init(global_allocator, 2);
    const key = try global_allocator.dupe(u8, sid);
    errdefer { limiter.deinit(); global_allocator.free(key); }
    try session_exa_limiters_map.put(key, limiter);
    return limiter;
}

fn evictSessions() void {
    const now = nowSeconds();

    var locked_sids = std.ArrayList([]u8).init(global_allocator);
    defer {
        for (locked_sids.items) |s| global_allocator.free(s);
        locked_sids.deinit();
    }

    {
        session_locks_guard.lock();
        defer session_locks_guard.unlock();
        var it = session_locks_map.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.*.tryLock()) {
                entry.value_ptr.*.unlock();
            } else {
                const copy = global_allocator.dupe(u8, entry.key_ptr.*) catch continue;
                locked_sids.append(copy) catch { global_allocator.free(copy); };
            }
        }
    }

    var evicted_sids = std.ArrayList([]u8).init(global_allocator);
    defer {
        for (evicted_sids.items) |s| global_allocator.free(s);
        evicted_sids.deinit();
    }

    var to_remove = std.ArrayList([]const u8).init(global_allocator);
    defer to_remove.deinit();

    {
        sessions_mutex.lock();
        defer sessions_mutex.unlock();

        var it = sessions_map.iterator();
        while (it.next()) |entry| {
            if (now - entry.value_ptr.*.updated_at > SESSION_TTL) {
                var is_locked = false;
                for (locked_sids.items) |ls| {
                    if (std.mem.eql(u8, ls, entry.key_ptr.*)) { is_locked = true; break; }
                }
                if (!is_locked) to_remove.append(entry.key_ptr.*) catch {};
            }
        }
        for (to_remove.items) |key| {
            if (sessions_map.fetchRemove(key)) |kv| {
                var sess = kv.value;
                sess.deinit(global_allocator);
                if (global_allocator.dupe(u8, kv.key)) |d| {
                    evicted_sids.append(d) catch global_allocator.free(d);
                } else |_| {}
                global_allocator.free(kv.key);
            }
        }
        to_remove.clearRetainingCapacity();

        if (sessions_map.count() > MAX_SESSIONS) {
            const SessionEntry = struct { key: []const u8, updated_at: f64 };
            var entries = std.ArrayList(SessionEntry).init(global_allocator);
            defer entries.deinit();
            var it2 = sessions_map.iterator();
            while (it2.next()) |entry| {
                var is_locked = false;
                for (locked_sids.items) |ls| {
                    if (std.mem.eql(u8, ls, entry.key_ptr.*)) { is_locked = true; break; }
                }
                if (!is_locked) entries.append(.{ .key = entry.key_ptr.*, .updated_at = entry.value_ptr.*.updated_at }) catch {};
            }
            std.sort.block(SessionEntry, entries.items, {}, struct {
                fn lessThan(_: void, a: SessionEntry, b: SessionEntry) bool { return a.updated_at < b.updated_at; }
            }.lessThan);
            const overflow = sessions_map.count() - MAX_SESSIONS;
            for (entries.items[0..@min(overflow, entries.items.len)]) |e| {
                if (sessions_map.fetchRemove(e.key)) |kv| {
                    var sess = kv.value;
                    sess.deinit(global_allocator);
                    if (global_allocator.dupe(u8, kv.key)) |d| {
                        evicted_sids.append(d) catch global_allocator.free(d);
                    } else |_| {}
                    global_allocator.free(kv.key);
                }
            }
        }
    }

    if (evicted_sids.items.len == 0) return;

    {
        session_exa_limiters_mutex.lock();
        defer session_exa_limiters_mutex.unlock();
        for (evicted_sids.items) |sid| {
            if (session_exa_limiters_map.fetchRemove(sid)) |kv| {
                kv.value.deinit();
                global_allocator.destroy(kv.value);
                global_allocator.free(kv.key);
            }
        }
    }
}

pub fn saveSessionsToDisk() void {
    sessions_disk_mutex.lock();
    defer sessions_disk_mutex.unlock();

    const SessSnap = struct { sid: []u8, messages_json: []u8, created_at: f64, updated_at: f64 };
    var snaps = std.ArrayList(SessSnap).init(global_allocator);
    defer {
        for (snaps.items) |s| {
            global_allocator.free(s.sid);
            global_allocator.free(s.messages_json);
        }
        snaps.deinit();
    }

    {
        sessions_mutex.lock();
        defer sessions_mutex.unlock();
        var it = sessions_map.iterator();
        while (it.next()) |entry| {
            const sid_copy = global_allocator.dupe(u8, entry.key_ptr.*) catch continue;
            var msg_buf = std.ArrayList(u8).init(global_allocator);
            serializeMessages(msg_buf.writer(), entry.value_ptr.*.messages.items) catch {
                msg_buf.deinit();
                global_allocator.free(sid_copy);
                continue;
            };
            const msg_json = msg_buf.toOwnedSlice() catch {
                msg_buf.deinit();
                global_allocator.free(sid_copy);
                continue;
            };
            snaps.append(.{
                .sid = sid_copy,
                .messages_json = msg_json,
                .created_at = entry.value_ptr.*.created_at,
                .updated_at = entry.value_ptr.*.updated_at,
            }) catch {
                global_allocator.free(sid_copy);
                global_allocator.free(msg_json);
                continue;
            };
        }
    }

    var json_buf = std.ArrayList(u8).init(global_allocator);
    defer json_buf.deinit();
    var w = json_buf.writer();
    w.writeByte('{') catch return;
    for (snaps.items, 0..) |snap, i| {
        if (i > 0) w.writeByte(',') catch return;
        writeJsonString(w, snap.sid) catch return;
        w.writeByte(':') catch return;
        w.writeByte('{') catch return;
        w.writeAll("\"messages\":") catch return;
        w.writeAll(snap.messages_json) catch return;
        w.writeAll(",\"created_at\":") catch return;
        std.fmt.format(w, "{d}", .{snap.created_at}) catch return;
        w.writeAll(",\"updated_at\":") catch return;
        std.fmt.format(w, "{d}", .{snap.updated_at}) catch return;
        w.writeByte('}') catch return;
    }
    w.writeByte('}') catch return;

    const nano = std.time.nanoTimestamp();
    const tmp_file = std.fmt.allocPrint(global_allocator, "{s}.{d}.tmp", .{ persistence_file, nano }) catch return;
    defer global_allocator.free(tmp_file);
    const f = std.fs.createFileAbsolute(tmp_file, .{}) catch return;
    f.writeAll(json_buf.items) catch {
        f.close();
        std.fs.deleteFileAbsolute(tmp_file) catch {};
        return;
    };
    f.close();
    std.fs.renameAbsolute(tmp_file, persistence_file) catch {
        std.fs.deleteFileAbsolute(tmp_file) catch {};
    };
}

pub fn errorToHungarian(err: anyerror) []const u8 {
    return switch (err) {
        error.CurlFailed => "Hálózati hiba a felső szolgáltatóval. Próbáld újra néhány másodperc múlva.",
        error.ExaApiKeyNotConfigured => "A keresőszolgáltatás nincs beállítva.",
        error.EmptyQuery => "Üres keresési lekérdezés.",
        error.ExaRateLimited => "A keresőszolgáltatás kérési korlátba ütközött. Várj egy kicsit.",
        error.ExaServerError => "A keresőszolgáltatás hibát adott vissza. Próbáld újra.",
        error.ExaInvalidResponse => "A keresőszolgáltatás érvénytelen választ adott.",
        error.ExaFailed => "A keresés meghiúsult. Próbáld újra.",
        error.ImageTooLarge => "A kép túl nagy.",
        error.UnsupportedImageFormat => "Nem támogatott képformátum.",
        error.ImageMediaTypeMismatch => "A kép típusa nem egyezik a tartalommal.",
        error.InvalidBase64, error.EmptyBase64, error.EmptyBase64AfterStrip, error.InvalidBase64Encoding => "Érvénytelen kép kódolás.",
        error.OutOfMemory => "A szerver memóriája megtelt. Próbáld újra.",
        error.SessionDeleted => "A munkamenet törölve lett.",
        else => "",
    };
}

pub fn sendErrorEvent(allocator: std.mem.Allocator, writer: anytype, err: anyerror) !void {
    const friendly = errorToHungarian(err);
    var ev_buf = std.ArrayList(u8).init(allocator);
    defer ev_buf.deinit();
    const w = ev_buf.writer();
    try w.writeAll("{\"type\":\"error\",\"message\":");
    if (friendly.len > 0) {
        try writeJsonString(w, friendly);
    } else {
        const fallback = try std.fmt.allocPrint(allocator, "Szerverhiba: {s}", .{@errorName(err)});
        defer allocator.free(fallback);
        try writeJsonString(w, fallback);
    }
    try w.writeAll(",\"code\":");
    try writeJsonString(w, @errorName(err));
    try w.writeByte('}');
    try sendSseEvent(writer, ev_buf.items);
}

pub fn writeJsonString(writer: anytype, s: []const u8) !void {
    try writer.writeByte('"');
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        switch (c) {
            '"' => { try writer.writeAll("\\\""); i += 1; },
            '\\' => { try writer.writeAll("\\\\"); i += 1; },
            '\n' => { try writer.writeAll("\\n"); i += 1; },
            '\r' => { try writer.writeAll("\\r"); i += 1; },
            '\t' => { try writer.writeAll("\\t"); i += 1; },
            0x00...0x08, 0x0b, 0x0c, 0x0e...0x1f => { try std.fmt.format(writer, "\\u{x:0>4}", .{c}); i += 1; },
            0x80...0xBF => { try writer.writeAll("\\ufffd"); i += 1; },
            0xC0...0xC1, 0xF5...0xFF => { try writer.writeAll("\\ufffd"); i += 1; },
            0xC2...0xDF => {
                if (i + 1 < s.len and (s[i + 1] & 0xC0) == 0x80) {
                    try writer.writeByte(c);
                    try writer.writeByte(s[i + 1]);
                    i += 2;
                } else { try writer.writeAll("\\ufffd"); i += 1; }
            },
            0xE0...0xEF => {
                if (i + 2 < s.len and (s[i + 1] & 0xC0) == 0x80 and (s[i + 2] & 0xC0) == 0x80) {
                    try writer.writeByte(c);
                    try writer.writeByte(s[i + 1]);
                    try writer.writeByte(s[i + 2]);
                    i += 3;
                } else { try writer.writeAll("\\ufffd"); i += 1; }
            },
            0xF0...0xF4 => {
                if (i + 3 < s.len and (s[i + 1] & 0xC0) == 0x80 and (s[i + 2] & 0xC0) == 0x80 and (s[i + 3] & 0xC0) == 0x80) {
                    try writer.writeByte(c);
                    try writer.writeByte(s[i + 1]);
                    try writer.writeByte(s[i + 2]);
                    try writer.writeByte(s[i + 3]);
                    i += 4;
                } else { try writer.writeAll("\\ufffd"); i += 1; }
            },
            else => { try writer.writeByte(c); i += 1; },
        }
    }
    try writer.writeByte('"');
}

pub fn serializeMessages(writer: anytype, messages: []const Message) !void {
    try writer.writeByte('[');
    for (messages, 0..) |msg, i| {
        if (i > 0) try writer.writeByte(',');
        try serializeMessage(writer, msg);
    }
    try writer.writeByte(']');
}

pub fn serializeMessage(writer: anytype, msg: Message) !void {
    try writer.writeByte('{');
    try writer.writeAll("\"role\":");
    try writeJsonString(writer, msg.role);
    try writer.writeAll(",\"content\":");
    switch (msg.content) {
        .text => |t| try writeJsonString(writer, t),
        .parts => |parts| {
            try writer.writeByte('[');
            for (parts.items, 0..) |part, i| {
                if (i > 0) try writer.writeByte(',');
                try serializeContentPart(writer, part);
            }
            try writer.writeByte(']');
        },
    }
    if (msg.tool_calls) |tcs| {
        try writer.writeAll(",\"tool_calls\":[");
        for (tcs.items, 0..) |tc, i| {
            if (i > 0) try writer.writeByte(',');
            try writer.writeAll("{\"id\":");
            try writeJsonString(writer, tc.id);
            try writer.writeAll(",\"type\":");
            try writeJsonString(writer, tc.type);
            try writer.writeAll(",\"function\":{\"name\":");
            try writeJsonString(writer, tc.function.name);
            try writer.writeAll(",\"arguments\":");
            try writeJsonString(writer, tc.function.arguments);
            try writer.writeAll("}}");
        }
        try writer.writeByte(']');
    }
    if (msg.tool_call_id) |id| { try writer.writeAll(",\"tool_call_id\":"); try writeJsonString(writer, id); }
    if (msg.msg_id) |mid| { try writer.writeAll(",\"msg_id\":"); try writeJsonString(writer, mid); }
    try writer.writeByte('}');
}

pub fn serializeContentPart(writer: anytype, part: MessageContentPart) !void {
    try writer.writeByte('{');
    try writer.writeAll("\"type\":");
    try writeJsonString(writer, part.type);
    if (part.text) |t| { try writer.writeAll(",\"text\":"); try writeJsonString(writer, t); }
    if (part.media_type) |m| { try writer.writeAll(",\"media_type\":"); try writeJsonString(writer, m); }
    if (part.data) |d| { try writer.writeAll(",\"data\":"); try writeJsonString(writer, d); }
    if (part.key) |k| { try writer.writeAll(",\"key\":"); try writeJsonString(writer, k); }
    if (part.image_url) |iu| {
        try writer.writeAll(",\"image_url\":{\"url\":");
        try writeJsonString(writer, iu.url);
        try writer.writeAll(",\"detail\":");
        try writeJsonString(writer, iu.detail);
        try writer.writeByte('}');
    }
    try writer.writeByte('}');
}

fn loadSessionsFromDisk() void {
    const f = std.fs.openFileAbsolute(persistence_file, .{}) catch return;
    defer f.close();
    const content = f.readToEndAlloc(global_allocator, 100 * 1024 * 1024) catch return;
    defer global_allocator.free(content);

    const parsed = std.json.parseFromSlice(std.json.Value, global_allocator, content, .{}) catch return;
    defer parsed.deinit();

    if (parsed.value != .object) return;

    const now = nowSeconds();
    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        {
            sessions_mutex.lock();
            const count = sessions_map.count();
            sessions_mutex.unlock();
            if (count >= MAX_SESSIONS) break;
        }

        const k = entry.key_ptr.*;
        const v = entry.value_ptr.*;
        if (v != .object) continue;
        const msgs_val = v.object.get("messages") orelse continue;
        const updated_val = v.object.get("updated_at") orelse continue;
        const created_val = v.object.get("created_at") orelse v.object.get("updated_at") orelse continue;
        if (msgs_val != .array) continue;

        const updated_at = jsonValueToFloat(updated_val);
        if (now - updated_at > SESSION_TTL) continue;

        const created_at = jsonValueToFloat(created_val);

        var sess = Session.init(global_allocator, created_at);
        sess.updated_at = updated_at;

        var msg_count: usize = 0;
        for (msgs_val.array.items) |msg_val| {
            if (msg_count >= MAX_STORED_MESSAGES) break;
            if (msg_val != .object) continue;
            const msg = parseMessageFromJson(msg_val) catch continue;
            sess.messages.append(msg) catch { var m = msg; m.deinit(global_allocator); continue; };
            msg_count += 1;
        }

        const key = global_allocator.dupe(u8, k) catch { sess.deinit(global_allocator); continue; };
        sessions_mutex.lock();
        sessions_map.put(key, sess) catch {
            sessions_mutex.unlock();
            sess.deinit(global_allocator);
            global_allocator.free(key);
            continue;
        };
        sessions_mutex.unlock();
    }
}

fn jsonValueToFloat(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch 0.0,
        else => 0.0,
    };
}

fn jsonValueToString(allocator: std.mem.Allocator, v: std.json.Value) ![]u8 {
    return switch (v) {
        .string => |s| allocator.dupe(u8, s),
        .integer => |i| std.fmt.allocPrint(allocator, "{d}", .{i}),
        .float => |f| std.fmt.allocPrint(allocator, "{d}", .{f}),
        .bool => |b| allocator.dupe(u8, if (b) "true" else "false"),
        .null => allocator.dupe(u8, "null"),
        else => error.UnsupportedJsonType,
    };
}

fn parseMessageFromJson(v: std.json.Value) !Message {
    const obj = v.object;
    const role_val = obj.get("role") orelse return error.MissingRole;
    const role = try global_allocator.dupe(u8, if (role_val == .string) role_val.string else "user");
    errdefer global_allocator.free(role);

    const content_val = obj.get("content") orelse std.json.Value{ .string = "" };
    var content: MessageContent = undefined;

    if (content_val == .string) {
        content = MessageContent{ .text = try global_allocator.dupe(u8, content_val.string) };
    } else if (content_val == .array) {
        var parts = std.ArrayList(MessageContentPart).init(global_allocator);
        errdefer {
            for (parts.items) |*p| p.deinit(global_allocator);
            parts.deinit();
        }
        for (content_val.array.items) |part_val| {
            if (part_val != .object) continue;
            const part = try parseContentPartFromJson(part_val);
            parts.append(part) catch |err| {
                var p = part;
                p.deinit(global_allocator);
                return err;
            };
        }
        content = MessageContent{ .parts = parts };
    } else {
        content = MessageContent{ .text = try global_allocator.dupe(u8, "") };
    }
    errdefer content.deinit(global_allocator);

    var tool_calls: ?std.ArrayList(ToolCall) = null;
    if (obj.get("tool_calls")) |tcs_val| {
        if (tcs_val == .array) {
            var tcs = std.ArrayList(ToolCall).init(global_allocator);
            errdefer {
                for (tcs.items) |*tc| tc.deinit(global_allocator);
                tcs.deinit();
            }
            for (tcs_val.array.items) |tc_val| {
                if (tc_val != .object) continue;
                const tc = try parseToolCallFromJson(tc_val);
                tcs.append(tc) catch |err| {
                    var t = tc;
                    t.deinit(global_allocator);
                    return err;
                };
            }
            tool_calls = tcs;
        }
    }

    const tool_call_id = if (obj.get("tool_call_id")) |id_val| if (id_val == .string) try global_allocator.dupe(u8, id_val.string) else null else null;
    const msg_id = if (obj.get("msg_id")) |mid_val| if (mid_val == .string) try global_allocator.dupe(u8, mid_val.string) else null else null;

    return Message{
        .role = role, .content = content, .tool_calls = tool_calls,
        .tool_call_id = tool_call_id, .msg_id = msg_id,
        .cached_size = null,
    };
}

fn parseContentPartFromJson(v: std.json.Value) !MessageContentPart {
    const obj = v.object;
    const type_val = obj.get("type") orelse return error.MissingType;
    const part_type = try global_allocator.dupe(u8, if (type_val == .string) type_val.string else "text");
    errdefer global_allocator.free(part_type);

    const text: ?[]u8 = if (obj.get("text")) |t| if (t == .string) try global_allocator.dupe(u8, t.string) else null else null;
    errdefer if (text) |t| global_allocator.free(t);

    const media_type: ?[]u8 = if (obj.get("media_type")) |m| if (m == .string) try global_allocator.dupe(u8, m.string) else null else null;
    errdefer if (media_type) |m| global_allocator.free(m);

    const data: ?[]u8 = if (obj.get("data")) |d| if (d == .string) try global_allocator.dupe(u8, d.string) else null else null;
    errdefer if (data) |d| global_allocator.free(d);

    const key: ?[]u8 = if (obj.get("key")) |k| if (k == .string) try global_allocator.dupe(u8, k.string) else null else null;
    errdefer if (key) |k| global_allocator.free(k);

    var image_url: ?MessageContentPart.ImageUrl = null;
    if (obj.get("image_url")) |iu_val| {
        if (iu_val == .object) {
            const url_val = iu_val.object.get("url") orelse std.json.Value{ .string = "" };
            const detail_val = iu_val.object.get("detail") orelse std.json.Value{ .string = "" };
            const url = try global_allocator.dupe(u8, if (url_val == .string) url_val.string else "");
            errdefer global_allocator.free(url);
            const detail = try global_allocator.dupe(u8, if (detail_val == .string) detail_val.string else "");
            image_url = .{ .url = url, .detail = detail };
        }
    }

    return MessageContentPart{ .type = part_type, .text = text, .media_type = media_type, .data = data, .key = key, .image_url = image_url };
}

fn parseToolCallFromJson(v: std.json.Value) !ToolCall {
    const obj = v.object;
    const id_val = obj.get("id") orelse std.json.Value{ .string = "" };
    const type_val = obj.get("type") orelse std.json.Value{ .string = "function" };
    const func_val = obj.get("function");

    var name: []u8 = try global_allocator.dupe(u8, "");
    errdefer global_allocator.free(name);
    var arguments: []u8 = try global_allocator.dupe(u8, "{}");
    errdefer global_allocator.free(arguments);

    if (func_val) |fv| {
        if (fv == .object) {
            if (fv.object.get("name")) |n| {
                if (n == .string) {
                    const new_name = try global_allocator.dupe(u8, n.string);
                    global_allocator.free(name);
                    name = new_name;
                }
            }
            if (fv.object.get("arguments")) |a| {
                if (a == .string) {
                    const new_args = try global_allocator.dupe(u8, a.string);
                    global_allocator.free(arguments);
                    arguments = new_args;
                }
            }
        }
    }

    const id = try global_allocator.dupe(u8, if (id_val == .string) id_val.string else "");
    errdefer global_allocator.free(id);
    const tc_type = try global_allocator.dupe(u8, if (type_val == .string) type_val.string else "function");

    return ToolCall{
        .id = id,
        .type = tc_type,
        .function = .{ .name = name, .arguments = arguments },
    };
}

pub fn messageCharSize(msg: *Message) usize {
    if (msg.cached_size) |cs| return cs;
    var total: usize = msg.role.len + 32;
    switch (msg.content) {
        .text => |t| total += t.len,
        .parts => |parts| {
            for (parts.items) |part| {
                const ptype = part.type;
                if (std.mem.eql(u8, ptype, "text")) {
                    total += if (part.text) |t| t.len else 0;
                } else if (std.mem.eql(u8, ptype, "image_url")) {
                    if (part.image_url) |iu| { total += iu.url.len + iu.detail.len + 32; } else { total += 32; }
                } else if (std.mem.eql(u8, ptype, "image_ref")) {
                    total += (if (part.key) |k| k.len else 0) + (if (part.media_type) |m| m.len else 0) + 32;
                } else if (std.mem.eql(u8, ptype, "image_inline")) {
                    total += (if (part.data) |d| d.len else 0) + (if (part.media_type) |m| m.len else 0) + 32;
                } else { total += 64; }
            }
        },
    }
    if (msg.tool_calls) |tcs| { for (tcs.items) |tc| total += tc.id.len + tc.type.len + tc.function.name.len + tc.function.arguments.len + 32; }
    if (msg.tool_call_id) |id| total += id.len;
    if (msg.msg_id) |mid| total += mid.len;
    msg.cached_size = total;
    return total;
}

pub fn safeMaxTokens(messages: []Message) usize {
    var total_chars: usize = 0;
    for (messages) |*msg| total_chars += messageCharSize(@constCast(msg));
    const estimated_input_tokens = (total_chars / 3) + 2000;
    if (estimated_input_tokens >= MODEL_CONTEXT_LIMIT) return 1;
    return @min(DESIRED_MAX_TOKENS, MODEL_CONTEXT_LIMIT - estimated_input_tokens);
}

fn findToolBoundary(messages: []const Message, start_idx: usize) usize {
    var idx = start_idx;
    while (idx < messages.len) {
        const m = messages[idx];
        if (std.mem.eql(u8, m.role, "tool")) { idx += 1; continue; }
        if (std.mem.eql(u8, m.role, "assistant") and m.tool_calls != null and idx + 1 < messages.len) {
            var j = idx + 1;
            while (j < messages.len and std.mem.eql(u8, messages[j].role, "tool")) j += 1;
            if (j > idx + 1) { idx = j; continue; }
        }
        break;
    }
    return idx;
}

fn findTurnBoundary(messages: []const Message, min_idx: usize) usize {
    var idx = findToolBoundary(messages, min_idx);
    while (idx < messages.len and !std.mem.eql(u8, messages[idx].role, "user")) idx += 1;
    if (idx >= messages.len and messages.len > 0) {
        var fallback = findToolBoundary(messages, min_idx);
        while (fallback < messages.len and std.mem.eql(u8, messages[fallback].role, "tool")) fallback += 1;
        idx = @min(fallback, messages.len);
    }
    return @min(idx, messages.len);
}

fn stripOrphanedAssistantToolCalls(allocator: std.mem.Allocator, messages: *std.ArrayList(Message)) !void {
    if (messages.items.len == 0) return;

    var tool_result_ids = std.StringHashMap(void).init(allocator);
    defer tool_result_ids.deinit();
    var asst_tool_call_ids = std.StringHashMap(void).init(allocator);
    defer asst_tool_call_ids.deinit();

    for (messages.items) |msg| {
        if (std.mem.eql(u8, msg.role, "tool")) {
            if (msg.tool_call_id) |id| try tool_result_ids.put(id, {});
        }
        if (std.mem.eql(u8, msg.role, "assistant")) {
            if (msg.tool_calls) |tcs| {
                for (tcs.items) |tc| try asst_tool_call_ids.put(tc.id, {});
            }
        }
    }

    var cleaned = std.ArrayList(Message).init(allocator);
    errdefer {
        for (cleaned.items) |*m| m.deinit(allocator);
        cleaned.deinit();
    }

    for (messages.items) |*msg| {
        if (std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
            var remaining_tcs = std.ArrayList(ToolCall).init(allocator);
            errdefer {
                for (remaining_tcs.items) |*tc| tc.deinit(allocator);
                remaining_tcs.deinit();
            }
            for (msg.tool_calls.?.items) |tc| {
                if (tool_result_ids.get(tc.id) != null) try remaining_tcs.append(try tc.clone(allocator));
            }

            if (remaining_tcs.items.len > 0) {
                var new_msg = try msg.clone(allocator);
                errdefer new_msg.deinit(allocator);
                if (new_msg.tool_calls) |*old_tcs| {
                    for (old_tcs.items) |*tc| tc.deinit(allocator);
                    old_tcs.deinit();
                }
                new_msg.tool_calls = remaining_tcs;
                try cleaned.append(new_msg);
            } else {
                remaining_tcs.deinit();
                var text_content: []const u8 = "";
                switch (msg.content) {
                    .text => |t| text_content = t,
                    .parts => |parts| {
                        for (parts.items) |part| {
                            if (std.mem.eql(u8, part.type, "text")) {
                                if (part.text) |t| { text_content = t; break; }
                            }
                        }
                    },
                }
                const role_copy = try allocator.dupe(u8, msg.role);
                errdefer allocator.free(role_copy);
                const text_val = if (text_content.len > 0) text_content else " ";
                const text_copy = try allocator.dupe(u8, text_val);
                errdefer allocator.free(text_copy);
                const mid_copy: ?[]u8 = if (msg.msg_id) |mid| try allocator.dupe(u8, mid) else null;
                try cleaned.append(Message{
                    .role = role_copy,
                    .content = MessageContent{ .text = text_copy },
                    .tool_calls = null, .tool_call_id = null,
                    .msg_id = mid_copy,
                    .cached_size = null,
                });
            }
        } else if (std.mem.eql(u8, msg.role, "tool")) {
            if (msg.tool_call_id) |id| {
                if (asst_tool_call_ids.get(id) != null) {
                    try cleaned.append(try msg.clone(allocator));
                }
            }
        } else {
            try cleaned.append(try msg.clone(allocator));
        }
    }

    for (messages.items) |*m| m.deinit(allocator);
    messages.deinit();
    messages.* = cleaned;
}

fn safeReplaceMessages(allocator: std.mem.Allocator, history: *std.ArrayList(Message), cut: usize) !void {
    if (cut == 0 or cut >= history.items.len) return;
    var new_list = std.ArrayList(Message).init(allocator);
    try new_list.ensureTotalCapacity(history.items.len - cut);
    for (history.items[cut..]) |msg| new_list.appendAssumeCapacity(msg);
    for (history.items[0..cut]) |*msg| msg.deinit(allocator);
    history.deinit();
    history.* = new_list;
}

fn trimHistory(allocator: std.mem.Allocator, history: *std.ArrayList(Message)) !void {
    if (history.items.len <= MAX_STORED_MESSAGES) return;

    const cut_raw = history.items.len - MAX_STORED_MESSAGES;
    const cut_boundary = findTurnBoundary(history.items, cut_raw);
    const actual_cut = if (cut_boundary >= history.items.len) cut_raw else cut_boundary;
    try safeReplaceMessages(allocator, history, actual_cut);

    try stripOrphanedAssistantToolCalls(allocator, history);

    var total_chars: usize = 0;
    for (history.items) |*msg| total_chars += messageCharSize(msg);
    while (total_chars > MAX_MODEL_CONTENT_CHARS * 2 and history.items.len > 1) {
        const cut2_raw = findTurnBoundary(history.items, 1);
        const cut2 = if (cut2_raw == 0 or cut2_raw >= history.items.len) 1 else cut2_raw;
        try safeReplaceMessages(allocator, history, cut2);
        total_chars = 0;
        for (history.items) |*msg| total_chars += messageCharSize(msg);
    }
}

pub fn buildApiMessages(allocator: std.mem.Allocator, history: []const Message) !std.ArrayList(Message) {
    var result = std.ArrayList(Message).init(allocator);
    errdefer {
        for (result.items) |*msg| msg.deinit(allocator);
        result.deinit();
    }
    for (history) |msg| try result.append(try msg.clone(allocator));

    try stripOrphanedAssistantToolCalls(allocator, &result);

    if (result.items.len > MAX_HISTORY_MESSAGES) {
        const cut_raw = result.items.len - MAX_HISTORY_MESSAGES;
        const cut_boundary = findTurnBoundary(result.items, cut_raw);
        const actual_cut = if (cut_boundary >= result.items.len) cut_raw else cut_boundary;
        try safeReplaceMessages(allocator, &result, actual_cut);
        if (result.items.len == 0) return result;
        try stripOrphanedAssistantToolCalls(allocator, &result);
    }

    var total_chars: usize = 0;
    for (result.items) |*msg| total_chars += messageCharSize(msg);

    while (result.items.len > 1 and total_chars > MAX_MODEL_CONTENT_CHARS) {
        const cut_raw = findTurnBoundary(result.items, 1);
        const cut = if (cut_raw == 0 or cut_raw >= result.items.len) 1 else cut_raw;
        try safeReplaceMessages(allocator, &result, cut);
        if (result.items.len == 0) return result;
        try stripOrphanedAssistantToolCalls(allocator, &result);
        total_chars = 0;
        for (result.items) |*msg| total_chars += messageCharSize(msg);
    }

    return result;
}

pub fn rollbackSessionTurn(sid: []const u8, msg_id: []const u8) void {
    sessions_mutex.lock();
    defer sessions_mutex.unlock();
    if (sessions_map.getPtr(sid)) |sess| {
        var entry_start: ?usize = null;
        for (sess.messages.items, 0..) |msg, i| {
            if (msg.msg_id) |mid| if (std.mem.eql(u8, mid, msg_id)) { entry_start = i; break; };
        }
        if (entry_start) |start| {
            for (sess.messages.items[start..]) |*msg| msg.deinit(global_allocator);
            sess.messages.shrinkRetainingCapacity(start);
            sess.updated_at = nowSeconds();
        }
    }
}

pub fn getSystemPrompt(allocator: std.mem.Allocator) ![]u8 {
    const now_ms = std.time.milliTimestamp();
    const now_sec = @divTrunc(now_ms, 1000);
    const epoch = std.time.epoch.EpochSeconds{ .secs = @intCast(now_sec) };
    const epoch_day = epoch.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const total_minutes = @divTrunc(@mod(now_ms, 86400_000), 60000);
    const hour = @divTrunc(total_minutes, 60);
    const minute = @mod(total_minutes, 60);
    return std.fmt.allocPrint(allocator,
        "You are tiffytime, an intelligent AI assistant.\nCurrent date: {d:0>4}-{d:0>2}-{d:0>2} | Current time: {d:0>2}:{d:0>2} UTC\n\nSEARCH RULES:\n- Use exa_search when the question requires current information, recent events, news, prices, people, sports results, or anything that may have changed.\n- Do NOT search for simple greetings, general knowledge, math, or timeless facts.\n- Provide multiple queries in the 'queries' array for comprehensive coverage.\n\nANSWER RULES:\n- If you searched: base your answer EXCLUSIVELY on the search result summaries received. Do NOT use your internal knowledge.\n- If you did not search: answer directly from your knowledge.\n- Synthesize the summaries into a clear, well-structured answer.\n- Cite sources with URLs where relevant.\n- Match the user's language and tone.\n- Be concise for simple questions, thorough for complex ones.",
        .{ year_day.year, month_day.month.numeric(), month_day.day_index + 1, hour, minute },
    );
}

fn bytesToHex(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    const result = try allocator.alloc(u8, bytes.len * 2);
    for (bytes, 0..) |b, i| {
        _ = std.fmt.bufPrint(result[i * 2 .. i * 2 + 2], "{x:0>2}", .{b}) catch unreachable;
    }
    return result;
}

fn hexToBytes(allocator: std.mem.Allocator, hex: []const u8) ![]u8 {
    if (hex.len % 2 != 0) return error.InvalidHex;
    const result = try allocator.alloc(u8, hex.len / 2);
    for (0..hex.len / 2) |i| result[i] = std.fmt.parseInt(u8, hex[i * 2 .. i * 2 + 2], 16) catch return error.InvalidHex;
    return result;
}

fn base64UrlEncode(allocator: std.mem.Allocator, data: []const u8) ![]u8 {
    const encoder = std.base64.url_safe_no_pad.Encoder;
    const len = encoder.calcSize(data.len);
    const result = try allocator.alloc(u8, len);
    _ = encoder.encode(result, data);
    return result;
}

fn base64UrlDecode(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    const decoder = std.base64.url_safe_no_pad.Decoder;
    const len = try decoder.calcSizeForSlice(s);
    const result = try allocator.alloc(u8, len);
    errdefer allocator.free(result);
    try decoder.decode(result, s);
    return result;
}

fn hashPassword(allocator: std.mem.Allocator, password: []const u8) ![]u8 {
    var salt: [16]u8 = undefined;
    std.crypto.random.bytes(&salt);
    var dk: [32]u8 = undefined;
    try std.crypto.pwhash.pbkdf2(&dk, password, &salt, 100_000, std.crypto.auth.hmac.sha2.HmacSha256);
    const salt_hex = try bytesToHex(allocator, &salt);
    defer allocator.free(salt_hex);
    const dk_hex = try bytesToHex(allocator, &dk);
    defer allocator.free(dk_hex);
    return std.fmt.allocPrint(allocator, "pbkdf2$100000${s}${s}", .{ salt_hex, dk_hex });
}

fn verifyPassword(allocator: std.mem.Allocator, password: []const u8, hash: []const u8) bool {
    var parts = std.mem.splitScalar(u8, hash, '$');
    const algo = parts.next() orelse return false;
    if (!std.mem.eql(u8, algo, "pbkdf2")) return false;
    const rounds_str = parts.next() orelse return false;
    const rounds = std.fmt.parseInt(u32, rounds_str, 10) catch return false;
    const salt_hex = parts.next() orelse return false;
    const dk_hex = parts.next() orelse return false;
    if (parts.next() != null) return false;

    const salt = hexToBytes(allocator, salt_hex) catch return false;
    defer allocator.free(salt);
    const stored_dk = hexToBytes(allocator, dk_hex) catch return false;
    defer allocator.free(stored_dk);

    if (stored_dk.len != 32) return false;

    var dk: [32]u8 = undefined;
    std.crypto.pwhash.pbkdf2(&dk, password, salt, rounds, std.crypto.auth.hmac.sha2.HmacSha256) catch return false;

    return std.crypto.utils.timingSafeEql([32]u8, dk, stored_dk[0..32].*);
}

fn generateAuthToken(allocator: std.mem.Allocator) ![]u8 {
    var bytes: [32]u8 = undefined;
    std.crypto.random.bytes(&bytes);
    return bytesToHex(allocator, &bytes);
}

fn generateChallenge(allocator: std.mem.Allocator) ![]u8 {
    var bytes: [32]u8 = undefined;
    std.crypto.random.bytes(&bytes);
    return base64UrlEncode(allocator, &bytes);
}

fn pruneExpiredTokensUnlocked() void {
    const now = nowSeconds();
    var to_remove = std.ArrayList([]const u8).init(global_allocator);
    defer to_remove.deinit();
    var it = auth_tokens.iterator();
    while (it.next()) |entry| {
        if (now > entry.value_ptr.*.expires_at) to_remove.append(entry.key_ptr.*) catch {};
    }
    for (to_remove.items) |key| {
        if (auth_tokens.fetchRemove(key)) |kv| {
            var ti = kv.value;
            ti.deinit(global_allocator);
            global_allocator.free(kv.key);
        }
    }
}

fn pruneExpiredChallengesUnlocked() void {
    const now = nowSeconds();
    var to_remove = std.ArrayList([]const u8).init(global_allocator);
    defer to_remove.deinit();
    var it = auth_challenges.iterator();
    while (it.next()) |entry| {
        if (now - entry.value_ptr.*.created_at > AUTH_CHALLENGE_TTL) to_remove.append(entry.key_ptr.*) catch {};
    }
    for (to_remove.items) |key| {
        if (auth_challenges.fetchRemove(key)) |kv| {
            var ci = kv.value;
            ci.deinit(global_allocator);
            global_allocator.free(kv.key);
        }
    }
}

const CborReader = struct {
    data: []const u8,
    pos: usize,

    fn init(data: []const u8) CborReader {
        return .{ .data = data, .pos = 0 };
    }

    fn readByte(self: *CborReader) !u8 {
        if (self.pos >= self.data.len) return error.CborUnexpectedEof;
        const b = self.data[self.pos];
        self.pos += 1;
        return b;
    }

    fn readUintN(self: *CborReader, n: usize) !u64 {
        if (self.pos + n > self.data.len) return error.CborUnexpectedEof;
        var result: u64 = 0;
        for (self.data[self.pos .. self.pos + n]) |byte| result = (result << 8) | byte;
        self.pos += n;
        return result;
    }

    fn readHead(self: *CborReader) !struct { major: u3, val: u64 } {
        const b = try self.readByte();
        const major: u3 = @intCast(b >> 5);
        const add_info: u5 = @intCast(b & 0x1f);
        const val: u64 = switch (add_info) {
            0...23 => add_info,
            24 => try self.readUintN(1),
            25 => try self.readUintN(2),
            26 => try self.readUintN(4),
            27 => try self.readUintN(8),
            31 => return error.CborInvalidAdditionalInfo,
            else => return error.CborInvalidAdditionalInfo,
        };
        return .{ .major = major, .val = val };
    }

    fn readBytes(self: *CborReader) ![]const u8 {
        const head = try self.readHead();
        if (head.major != 2) return error.CborExpectedBytes;
        if (head.val > self.data.len) return error.CborUnexpectedEof;
        const len: usize = @intCast(head.val);
        if (self.pos + len > self.data.len) return error.CborUnexpectedEof;
        const result = self.data[self.pos .. self.pos + len];
        self.pos += len;
        return result;
    }

    fn readText(self: *CborReader) ![]const u8 {
        const head = try self.readHead();
        if (head.major != 3) return error.CborExpectedText;
        if (head.val > self.data.len) return error.CborUnexpectedEof;
        const len: usize = @intCast(head.val);
        if (self.pos + len > self.data.len) return error.CborUnexpectedEof;
        const result = self.data[self.pos .. self.pos + len];
        self.pos += len;
        return result;
    }

    fn readInt(self: *CborReader) !i64 {
        const head = try self.readHead();
        return switch (head.major) {
            0 => if (head.val > @as(u64, std.math.maxInt(i64))) error.CborIntegerOverflow else @as(i64, @intCast(head.val)),
            1 => if (head.val > @as(u64, std.math.maxInt(i64))) error.CborIntegerOverflow else -@as(i64, @intCast(head.val)) - 1,
            else => error.CborExpectedInt,
        };
    }

    fn skipValue(self: *CborReader) !void {
        var to_skip: usize = 1;
        var depth: usize = 0;
        const MAX_DEPTH: usize = 512;
        while (to_skip > 0) {
            if (depth > MAX_DEPTH) return error.CborNestingTooDeep;
            to_skip -= 1;
            depth += 1;
            const head = try self.readHead();
            switch (head.major) {
                0, 1, 7 => {},
                2, 3 => {
                    if (head.val > self.data.len) return error.CborUnexpectedEof;
                    const len: usize = @intCast(head.val);
                    if (self.pos + len > self.data.len) return error.CborUnexpectedEof;
                    self.pos += len;
                },
                4 => {
                    if (head.val > self.data.len) return error.CborUnexpectedEof;
                    to_skip += @intCast(head.val);
                },
                5 => {
                    if (head.val > self.data.len / 2 + 1) return error.CborUnexpectedEof;
                    to_skip += @intCast(head.val * 2);
                },
                6 => {
                    to_skip += 1;
                },
            }
        }
    }
};

fn parseAttestationObject(allocator: std.mem.Allocator, cbor_data: []const u8) !struct { auth_data: []u8 } {
    var reader = CborReader.init(cbor_data);
    const head = try reader.readHead();
    if (head.major != 5) return error.CborExpectedMap;
    const map_len: usize = @intCast(head.val);

    var auth_data: ?[]const u8 = null;

    for (0..map_len) |_| {
        const key_head = reader.readHead() catch { reader.skipValue() catch {}; continue; };
        if (key_head.major == 3) {
            if (key_head.val > reader.data.len) return error.CborUnexpectedEof;
            const key_len: usize = @intCast(key_head.val);
            if (reader.pos + key_len > reader.data.len) return error.CborUnexpectedEof;
            const key_str = reader.data[reader.pos .. reader.pos + key_len];
            reader.pos += key_len;
            if (std.mem.eql(u8, key_str, "authData")) {
                auth_data = try reader.readBytes();
            } else {
                reader.skipValue() catch {};
            }
        } else {
            const extra: usize = switch (key_head.major) {
                2, 3 => if (key_head.val <= reader.data.len) @intCast(key_head.val) else return error.CborUnexpectedEof,
                else => 0,
            };
            if (extra > 0) {
                if (reader.pos + extra > reader.data.len) return error.CborUnexpectedEof;
                reader.pos += extra;
            }
            reader.skipValue() catch {};
        }
    }

    const ad = auth_data orelse return error.MissingAuthData;
    return .{ .auth_data = try allocator.dupe(u8, ad) };
}

const AuthDataResult = struct {
    rp_id_hash: [32]u8,
    flags: u8,
    sign_count: u32,
    cred_id: []u8,
    cose_key: []u8,
    allocator: std.mem.Allocator,

    fn deinit(self: *AuthDataResult) void {
        self.allocator.free(self.cred_id);
        self.allocator.free(self.cose_key);
    }
};

fn parseAuthData(allocator: std.mem.Allocator, auth_data: []const u8) !AuthDataResult {
    if (auth_data.len < 37) return error.AuthDataTooShort;
    var rp_id_hash: [32]u8 = undefined;
    @memcpy(&rp_id_hash, auth_data[0..32]);
    const flags = auth_data[32];
    const sign_count = std.mem.readInt(u32, auth_data[33..37], .big);
    if (flags & 0x40 == 0) return error.NoAttestedCredData;
    if (auth_data.len < 37 + 18) return error.AuthDataTooShort;
    const cred_id_len = std.mem.readInt(u16, auth_data[53..55], .big);
    if (auth_data.len < 55 + cred_id_len) return error.AuthDataTooShort;
    const cred_id = auth_data[55 .. 55 + cred_id_len];
    const cose_key_data = auth_data[55 + cred_id_len ..];
    const cred_id_copy = try allocator.dupe(u8, cred_id);
    errdefer allocator.free(cred_id_copy);
    const cose_key_copy = try allocator.dupe(u8, cose_key_data);
    return AuthDataResult{
        .rp_id_hash = rp_id_hash, .flags = flags, .sign_count = sign_count,
        .cred_id = cred_id_copy, .cose_key = cose_key_copy, .allocator = allocator,
    };
}

fn parseCoseP256Key(cose_data: []const u8) ![65]u8 {
    var reader = CborReader.init(cose_data);
    const head = try reader.readHead();
    if (head.major != 5) return error.CborExpectedMap;
    const map_len: usize = @intCast(head.val);

    var x: ?[32]u8 = null;
    var y: ?[32]u8 = null;
    var kty: ?i64 = null;
    var alg: ?i64 = null;
    var crv: ?i64 = null;

    for (0..map_len) |_| {
        const saved_pos = reader.pos;
        const key_val = reader.readInt() catch {
            reader.pos = saved_pos;
            reader.skipValue() catch return error.CborParseError;
            reader.skipValue() catch return error.CborParseError;
            continue;
        };
        switch (key_val) {
            1 => {
                kty = reader.readInt() catch blk: {
                    reader.skipValue() catch {};
                    break :blk null;
                };
            },
            3 => {
                alg = reader.readInt() catch blk: {
                    reader.skipValue() catch {};
                    break :blk null;
                };
            },
            -1 => {
                crv = reader.readInt() catch blk: {
                    reader.skipValue() catch {};
                    break :blk null;
                };
            },
            -2 => {
                const xb = reader.readBytes() catch { reader.skipValue() catch {}; continue; };
                if (xb.len == 32) x = xb[0..32].*;
            },
            -3 => {
                const yb = reader.readBytes() catch { reader.skipValue() catch {}; continue; };
                if (yb.len == 32) y = yb[0..32].*;
            },
            else => { reader.skipValue() catch {}; },
        }
    }

    if (kty == null or kty.? != 2) return error.InvalidCoseKeyType;
    if (alg == null or alg.? != -7) return error.InvalidCoseAlgorithm;
    if (crv == null or crv.? != 1) return error.InvalidCoseCurve;

    const xv = x orelse return error.MissingX;
    const yv = y orelse return error.MissingY;

    var result: [65]u8 = undefined;
    result[0] = 0x04;
    @memcpy(result[1..33], &xv);
    @memcpy(result[33..65], &yv);
    return result;
}

const EcdsaP256Sha256 = std.crypto.sign.ecdsa.Ecdsa(std.crypto.ecc.P256, std.crypto.hash.sha2.Sha256);

fn verifyWebAuthnSignature(sig_der: []const u8, auth_data_bytes: []const u8, client_data_json: []const u8, pubkey_uncompressed: [65]u8) !void {
    var client_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(client_data_json, &client_hash, .{});

    var signed_data = try global_allocator.alloc(u8, auth_data_bytes.len + 32);
    defer global_allocator.free(signed_data);
    @memcpy(signed_data[0..auth_data_bytes.len], auth_data_bytes);
    @memcpy(signed_data[auth_data_bytes.len..], &client_hash);

    const pubkey = EcdsaP256Sha256.PublicKey.fromSec1(&pubkey_uncompressed) catch return error.InvalidPublicKey;
    const sig = EcdsaP256Sha256.Signature.fromDer(sig_der) catch return error.InvalidSignature;
    try sig.verify(signed_data, pubkey);
}

fn saveAuthToDisk() void {
    auth_disk_mutex.lock();
    defer auth_disk_mutex.unlock();

    var json_buf = std.ArrayList(u8).init(global_allocator);
    defer json_buf.deinit();
    const w = json_buf.writer();

    {
        auth_mutex.lock();
        defer auth_mutex.unlock();

        pruneExpiredTokensUnlocked();
        pruneExpiredChallengesUnlocked();

        w.writeByte('{') catch return;
        w.writeAll("\"users\":{") catch return;
        var first = true;
        var it = auth_users.iterator();
        while (it.next()) |entry| {
            if (!first) w.writeByte(',') catch return;
            first = false;
            writeJsonString(w, entry.key_ptr.*) catch return;
            w.writeByte(':') catch return;
            w.writeByte('{') catch return;
            w.writeAll("\"id\":") catch return;
            writeJsonString(w, entry.value_ptr.*.id) catch return;
            w.writeAll(",\"email\":") catch return;
            writeJsonString(w, entry.value_ptr.*.email) catch return;
            w.writeAll(",\"password_hash\":") catch return;
            writeJsonString(w, entry.value_ptr.*.password_hash) catch return;
            std.fmt.format(w, ",\"created_at\":{d}", .{entry.value_ptr.*.created_at}) catch return;
            w.writeAll(",\"passkeys\":[") catch return;
            for (entry.value_ptr.*.passkeys.items, 0..) |pk, i| {
                if (i > 0) w.writeByte(',') catch return;
                w.writeByte('{') catch return;
                w.writeAll("\"credential_id\":") catch return;
                writeJsonString(w, pk.credential_id) catch return;
                w.writeAll(",\"public_key\":") catch return;
                writeJsonString(w, pk.public_key) catch return;
                std.fmt.format(w, ",\"sign_count\":{d}", .{pk.sign_count}) catch return;
                w.writeAll(",\"rp_id\":") catch return;
                writeJsonString(w, pk.rp_id) catch return;
                w.writeByte('}') catch return;
            }
            w.writeByte(']') catch return;
            w.writeByte('}') catch return;
        }
        w.writeByte('}') catch return;
        w.writeAll(",\"tokens\":{") catch return;
        first = true;
        var it2 = auth_tokens.iterator();
        while (it2.next()) |entry| {
            if (!first) w.writeByte(',') catch return;
            first = false;
            writeJsonString(w, entry.key_ptr.*) catch return;
            w.writeByte(':') catch return;
            w.writeByte('{') catch return;
            w.writeAll("\"email\":") catch return;
            writeJsonString(w, entry.value_ptr.*.email) catch return;
            std.fmt.format(w, ",\"expires_at\":{d}", .{entry.value_ptr.*.expires_at}) catch return;
            w.writeByte('}') catch return;
        }
        w.writeByte('}') catch return;
        w.writeByte('}') catch return;
    }

    const nano = std.time.nanoTimestamp();
    const tmp = std.fmt.allocPrint(global_allocator, "{s}.{d}.tmp", .{ auth_file, nano }) catch return;
    defer global_allocator.free(tmp);
    const f = std.fs.createFileAbsolute(tmp, .{}) catch return;
    f.writeAll(json_buf.items) catch {
        f.close();
        std.fs.deleteFileAbsolute(tmp) catch {};
        return;
    };
    f.close();
    std.fs.renameAbsolute(tmp, auth_file) catch {
        std.fs.deleteFileAbsolute(tmp) catch {};
    };
}

fn addCredIdIndex(cred_id: []const u8, email: []const u8) void {
    cred_id_map_mutex.lock();
    defer cred_id_map_mutex.unlock();
    if (cred_id_to_email.contains(cred_id)) return;
    const key = global_allocator.dupe(u8, cred_id) catch return;
    const val = global_allocator.dupe(u8, email) catch { global_allocator.free(key); return; };
    cred_id_to_email.put(key, val) catch {
        global_allocator.free(key);
        global_allocator.free(val);
    };
}

fn loadAuthFromDisk() void {
    const f = std.fs.openFileAbsolute(auth_file, .{}) catch return;
    defer f.close();
    const content = f.readToEndAlloc(global_allocator, 50 * 1024 * 1024) catch return;
    defer global_allocator.free(content);

    const parsed = std.json.parseFromSlice(std.json.Value, global_allocator, content, .{}) catch return;
    defer parsed.deinit();
    if (parsed.value != .object) return;

    if (parsed.value.object.get("users")) |users_val| {
        if (users_val == .object) {
            var it = users_val.object.iterator();
            while (it.next()) |entry| {
                const email_key = entry.key_ptr.*;
                const uobj = entry.value_ptr.*;
                if (uobj != .object) continue;

                const id_v = uobj.object.get("id") orelse continue;
                const email_v = uobj.object.get("email") orelse continue;
                const ph_v = uobj.object.get("password_hash") orelse std.json.Value{ .string = "" };
                const cat_v = uobj.object.get("created_at") orelse std.json.Value{ .float = 0.0 };

                const user_id = global_allocator.dupe(u8, if (id_v == .string) id_v.string else "") catch continue;
                errdefer_user_id: {
                    const user_email = global_allocator.dupe(u8, if (email_v == .string) email_v.string else "") catch {
                        global_allocator.free(user_id);
                        break :errdefer_user_id;
                    };
                    const user_ph = global_allocator.dupe(u8, if (ph_v == .string) ph_v.string else "") catch {
                        global_allocator.free(user_id);
                        global_allocator.free(user_email);
                        break :errdefer_user_id;
                    };

                    var user = AuthUser{
                        .id = user_id,
                        .email = user_email,
                        .password_hash = user_ph,
                        .passkeys = std.ArrayList(AuthPasskey).init(global_allocator),
                        .created_at = jsonValueToFloat(cat_v),
                    };

                    if (uobj.object.get("passkeys")) |pks_val| {
                        if (pks_val == .array) {
                            for (pks_val.array.items) |pk_val| {
                                if (pk_val != .object) continue;
                                const cid_v = pk_val.object.get("credential_id") orelse continue;
                                const pkey_v = pk_val.object.get("public_key") orelse continue;
                                const sc_v = pk_val.object.get("sign_count") orelse std.json.Value{ .integer = 0 };
                                const rpid_v = pk_val.object.get("rp_id") orelse std.json.Value{ .string = "" };

                                const pk_cid = global_allocator.dupe(u8, if (cid_v == .string) cid_v.string else "") catch continue;
                                const pk_pkey = global_allocator.dupe(u8, if (pkey_v == .string) pkey_v.string else "") catch {
                                    global_allocator.free(pk_cid);
                                    continue;
                                };
                                const pk_rp = global_allocator.dupe(u8, if (rpid_v == .string) rpid_v.string else "") catch {
                                    global_allocator.free(pk_cid);
                                    global_allocator.free(pk_pkey);
                                    continue;
                                };
                                const pk = AuthPasskey{
                                    .credential_id = pk_cid,
                                    .public_key = pk_pkey,
                                    .sign_count = if (sc_v == .integer) @intCast(sc_v.integer) else 0,
                                    .rp_id = pk_rp,
                                };
                                addCredIdIndex(pk_cid, email_key);
                                user.passkeys.append(pk) catch {
                                    var pk_mut = pk;
                                    pk_mut.deinit(global_allocator);
                                };
                            }
                        }
                    }

                    const key = global_allocator.dupe(u8, email_key) catch { user.deinit(global_allocator); break :errdefer_user_id; };
                    auth_users.put(key, user) catch { user.deinit(global_allocator); global_allocator.free(key); };
                }
            }
        }
    }

    if (parsed.value.object.get("tokens")) |tokens_val| {
        if (tokens_val == .object) {
            var it = tokens_val.object.iterator();
            while (it.next()) |entry| {
                const tok = entry.key_ptr.*;
                const tobj = entry.value_ptr.*;
                if (tobj != .object) continue;
                const email_v = tobj.object.get("email") orelse continue;
                const exp_v = tobj.object.get("expires_at") orelse continue;
                const exp = jsonValueToFloat(exp_v);
                if (nowSeconds() > exp) continue;
                const tok_email = global_allocator.dupe(u8, if (email_v == .string) email_v.string else "") catch continue;
                const info = AuthTokenInfo{ .email = tok_email, .expires_at = exp };
                const key = global_allocator.dupe(u8, tok) catch {
                    var ci = info;
                    ci.deinit(global_allocator);
                    continue;
                };
                auth_tokens.put(key, info) catch {
                    var ci = info;
                    ci.deinit(global_allocator);
                    global_allocator.free(key);
                };
            }
        }
    }
}

fn getRequestOrigin(headers: *const std.StringHashMap([]const u8)) []const u8 {
    if (headers.get("origin")) |o| return o;
    if (headers.get("referer")) |r| {
        if (std.mem.indexOf(u8, r, "//")) |dslash| {
            const after_scheme = r[dslash + 2 ..];
            const path_idx = std.mem.indexOf(u8, after_scheme, "/") orelse after_scheme.len;
            return r[0 .. dslash + 2 + path_idx];
        }
        return r;
    }
    return "https://localhost";
}

fn extractRpId(origin: []const u8) []const u8 {
    var s = origin;
    if (std.mem.startsWith(u8, s, "https://")) s = s[8..];
    if (std.mem.startsWith(u8, s, "http://")) s = s[7..];
    if (s.len > 0 and s[0] == '[') {
        if (std.mem.indexOf(u8, s, "]")) |end_bracket| {
            return s[0 .. end_bracket + 1];
        }
        return s;
    }
    if (std.mem.indexOf(u8, s, ":")) |colon| s = s[0..colon];
    if (std.mem.indexOf(u8, s, "/")) |slash| s = s[0..slash];
    return s;
}

fn getAuthEmailFromToken(allocator: std.mem.Allocator, headers: *const std.StringHashMap([]const u8)) ?[]u8 {
    const auth_header = headers.get("authorization") orelse return null;
    if (!std.mem.startsWith(u8, auth_header, "Bearer ")) return null;
    const token = auth_header[7..];
    auth_mutex.lock();
    defer auth_mutex.unlock();
    const tok_info = auth_tokens.get(token) orelse return null;
    if (nowSeconds() > tok_info.expires_at) {
        if (auth_tokens.fetchRemove(token)) |kv| {
            var ti = kv.value;
            ti.deinit(global_allocator);
            global_allocator.free(kv.key);
        }
        return null;
    }
    return allocator.dupe(u8, tok_info.email) catch null;
}

fn handleAuthRegister(allocator: std.mem.Allocator, writer: anytype, body: []const u8) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen kérés\"}");
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen kérés\"}");
        return;
    }

    const email_v = parsed.value.object.get("email") orelse { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"E-mail cím szükséges\"}"); return; };
    const pw_v = parsed.value.object.get("password") orelse { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Jelszó szükséges\"}"); return; };
    if (email_v != .string or pw_v != .string) { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen mezők\"}"); return; }

    const email = std.mem.trim(u8, email_v.string, " \t\n\r");
    const password = pw_v.string;
    if (email.len < 3 or !std.mem.containsAtLeast(u8, email, 1, "@")) { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen e-mail cím\"}"); return; }
    if (password.len < 8) { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"A jelszónak legalább 8 karakter hosszúnak kell lennie\"}"); return; }

    const hash = hashPassword(allocator, password) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
        return;
    };
    defer allocator.free(hash);
    const user_id = generateUuid(allocator) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
        return;
    };
    defer allocator.free(user_id);
    const token = generateAuthToken(allocator) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
        return;
    };
    defer allocator.free(token);

    var insert_ok = false;
    {
        auth_mutex.lock();
        defer auth_mutex.unlock();

        if (auth_users.get(email) != null) {
            try sendHttpResponse(writer, 409, "Conflict", "application/json", &[_][2][]const u8{}, "{\"error\":\"Ez az e-mail cím már regisztrált\"}");
            return;
        }

        const gid = global_allocator.dupe(u8, user_id) catch {
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        const gemail = global_allocator.dupe(u8, email) catch {
            global_allocator.free(gid);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        const ghash = global_allocator.dupe(u8, hash) catch {
            global_allocator.free(gid);
            global_allocator.free(gemail);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };

        const user = AuthUser{
            .id = gid,
            .email = gemail,
            .password_hash = ghash,
            .passkeys = std.ArrayList(AuthPasskey).init(global_allocator),
            .created_at = nowSeconds(),
        };
        const key = global_allocator.dupe(u8, email) catch {
            var u = user;
            u.deinit(global_allocator);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        auth_users.put(key, user) catch {
            var u = user;
            u.deinit(global_allocator);
            global_allocator.free(key);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };

        const tok_email = global_allocator.dupe(u8, email) catch {
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        const tok_info = AuthTokenInfo{ .email = tok_email, .expires_at = nowSeconds() + AUTH_TOKEN_TTL };
        const tok_key = global_allocator.dupe(u8, token) catch {
            var ti = tok_info;
            ti.deinit(global_allocator);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        auth_tokens.put(tok_key, tok_info) catch {
            var ti = tok_info;
            ti.deinit(global_allocator);
            global_allocator.free(tok_key);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        insert_ok = true;
    }

    if (!insert_ok) return;
    saveAuthToDisk();

    var resp_buf = std.ArrayList(u8).init(allocator);
    defer resp_buf.deinit();
    try resp_buf.writer().writeAll("{\"token\":");
    try writeJsonString(resp_buf.writer(), token);
    try resp_buf.writer().writeAll(",\"email\":");
    try writeJsonString(resp_buf.writer(), email);
    try resp_buf.writer().writeByte('}');
    try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, resp_buf.items);
}

fn handleAuthLogin(allocator: std.mem.Allocator, writer: anytype, body: []const u8) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen kérés\"}");
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen kérés\"}"); return; }

    const email_v = parsed.value.object.get("email") orelse { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"E-mail cím szükséges\"}"); return; };
    const pw_v = parsed.value.object.get("password") orelse { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Jelszó szükséges\"}"); return; };
    if (email_v != .string or pw_v != .string) { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen mezők\"}"); return; }

    const email = std.mem.trim(u8, email_v.string, " \t\n\r");
    const password = pw_v.string;

    var hash_copy: ?[]u8 = null;
    defer if (hash_copy) |h| allocator.free(h);
    var is_passkey_only = false;

    {
        auth_mutex.lock();
        defer auth_mutex.unlock();
        if (auth_users.get(email)) |user| {
            if (user.password_hash.len == 0) {
                is_passkey_only = true;
            } else {
                hash_copy = allocator.dupe(u8, user.password_hash) catch null;
            }
        }
    }

    if (hash_copy == null and !is_passkey_only) {
        try sendHttpResponse(writer, 401, "Unauthorized", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen e-mail cím vagy jelszó\"}");
        return;
    }

    if (is_passkey_only) {
        try sendHttpResponse(writer, 401, "Unauthorized", "application/json", &[_][2][]const u8{}, "{\"error\":\"Ez a fiók csak kulcsszóval rendelkezik, jelszóval nem lehet bejelentkezni\"}");
        return;
    }

    if (!verifyPassword(allocator, password, hash_copy.?)) {
        try sendHttpResponse(writer, 401, "Unauthorized", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen e-mail cím vagy jelszó\"}");
        return;
    }

    const token = generateAuthToken(allocator) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
        return;
    };
    defer allocator.free(token);

    {
        auth_mutex.lock();
        defer auth_mutex.unlock();
        const tok_email = global_allocator.dupe(u8, email) catch {
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        const tok_info = AuthTokenInfo{ .email = tok_email, .expires_at = nowSeconds() + AUTH_TOKEN_TTL };
        const tok_key = global_allocator.dupe(u8, token) catch {
            var ti = tok_info;
            ti.deinit(global_allocator);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        auth_tokens.put(tok_key, tok_info) catch {
            var ti = tok_info;
            ti.deinit(global_allocator);
            global_allocator.free(tok_key);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
    }

    saveAuthToDisk();

    var resp_buf = std.ArrayList(u8).init(allocator);
    defer resp_buf.deinit();
    try resp_buf.writer().writeAll("{\"token\":");
    try writeJsonString(resp_buf.writer(), token);
    try resp_buf.writer().writeAll(",\"email\":");
    try writeJsonString(resp_buf.writer(), email);
    try resp_buf.writer().writeByte('}');
    try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, resp_buf.items);
}

fn handleAuthMe(allocator: std.mem.Allocator, writer: anytype, headers: *const std.StringHashMap([]const u8)) !void {
    const email = getAuthEmailFromToken(allocator, headers) orelse {
        try sendHttpResponse(writer, 401, "Unauthorized", "application/json", &[_][2][]const u8{}, "{\"error\":\"Nincs bejelentkezve\"}");
        return;
    };
    defer allocator.free(email);
    var resp_buf = std.ArrayList(u8).init(allocator);
    defer resp_buf.deinit();
    try resp_buf.writer().writeAll("{\"email\":");
    try writeJsonString(resp_buf.writer(), email);
    try resp_buf.writer().writeByte('}');
    try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, resp_buf.items);
}

fn handleAuthLogout(writer: anytype, headers: *const std.StringHashMap([]const u8)) !void {
    const auth_header = headers.get("authorization") orelse {
        try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, "{\"ok\":true}");
        return;
    };
    if (std.mem.startsWith(u8, auth_header, "Bearer ")) {
        const token = auth_header[7..];
        auth_mutex.lock();
        defer auth_mutex.unlock();
        if (auth_tokens.fetchRemove(token)) |kv| {
            var ti = kv.value;
            ti.deinit(global_allocator);
            global_allocator.free(kv.key);
        }
    }
    saveAuthToDisk();
    try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, "{\"ok\":true}");
}

fn handleWebAuthnRegisterChallenge(allocator: std.mem.Allocator, writer: anytype, body: []const u8, headers: *const std.StringHashMap([]const u8)) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen kérés\"}");
        return;
    };
    defer parsed.deinit();

    var email: []const u8 = "";
    if (parsed.value == .object) {
        if (parsed.value.object.get("email")) |ev| if (ev == .string) { email = std.mem.trim(u8, ev.string, " \t\n\r"); };
    }

    var token_email: ?[]u8 = null;
    defer if (token_email) |te| allocator.free(te);

    if (email.len == 0) {
        token_email = getAuthEmailFromToken(allocator, headers);
        if (token_email) |te| email = te;
    }
    if (email.len == 0) {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"E-mail cím szükséges\"}");
        return;
    }

    const challenge = generateChallenge(allocator) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
        return;
    };
    defer allocator.free(challenge);

    const origin = getRequestOrigin(headers);
    const rp_id = extractRpId(origin);

    {
        auth_mutex.lock();
        defer auth_mutex.unlock();
        pruneExpiredChallengesUnlocked();

        const ch_email = global_allocator.dupe(u8, email) catch {
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        const ch_origin = global_allocator.dupe(u8, origin) catch {
            global_allocator.free(ch_email);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        const ch_info = AuthChallengeInfo{
            .email = ch_email,
            .is_login = false,
            .created_at = nowSeconds(),
            .origin = ch_origin,
        };
        const ch_key = global_allocator.dupe(u8, challenge) catch {
            var ci = ch_info;
            ci.deinit(global_allocator);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        auth_challenges.put(ch_key, ch_info) catch {
            var ci = ch_info;
            ci.deinit(global_allocator);
            global_allocator.free(ch_key);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
    }

    var user_id_b64: []u8 = undefined;
    var user_id_b64_owned = false;
    defer if (user_id_b64_owned) allocator.free(user_id_b64);

    {
        auth_mutex.lock();
        defer auth_mutex.unlock();
        if (auth_users.get(email)) |user| {
            user_id_b64 = base64UrlEncode(allocator, user.id) catch {
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            user_id_b64_owned = true;
        } else {
            var rand_id: [16]u8 = undefined;
            std.crypto.random.bytes(&rand_id);
            user_id_b64 = base64UrlEncode(allocator, &rand_id) catch {
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            user_id_b64_owned = true;
        }
    }

    var resp_buf = std.ArrayList(u8).init(allocator);
    defer resp_buf.deinit();
    const w = resp_buf.writer();
    try w.writeByte('{');
    try w.writeAll("\"challenge\":");
    try writeJsonString(w, challenge);
    try w.writeAll(",\"rpId\":");
    try writeJsonString(w, rp_id);
    try w.writeAll(",\"rpName\":\"tiffytime\",\"userId\":");
    try writeJsonString(w, user_id_b64);
    try w.writeAll(",\"userName\":");
    try writeJsonString(w, email);
    try w.writeAll(",\"userDisplayName\":");
    try writeJsonString(w, email);
    try w.writeByte('}');
    try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, resp_buf.items);
}

fn handleWebAuthnRegisterVerify(allocator: std.mem.Allocator, writer: anytype, body: []const u8) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen kérés\"}");
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen kérés\"}"); return; }
    const obj = parsed.value.object;

    const challenge_v = obj.get("challenge") orelse { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Hiányzó challenge\"}"); return; };
    const cdj_v = obj.get("clientDataJSON") orelse { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Hiányzó clientDataJSON\"}"); return; };
    const ao_v = obj.get("attestationObject") orelse { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Hiányzó attestationObject\"}"); return; };

    if (challenge_v != .string or cdj_v != .string or ao_v != .string) {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen mezők\"}");
        return;
    }

    const challenge = challenge_v.string;

    var ch_fetched: ?struct { info: AuthChallengeInfo, key: []const u8 } = null;
    defer if (ch_fetched) |*cf| {
        cf.info.deinit(global_allocator);
        global_allocator.free(cf.key);
    };

    {
        auth_mutex.lock();
        defer auth_mutex.unlock();
        pruneExpiredChallengesUnlocked();
        if (auth_challenges.fetchRemove(challenge)) |kv| {
            ch_fetched = .{ .info = kv.value, .key = kv.key };
        }
    }

    const ch_info = if (ch_fetched) |*cf| &cf.info else {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen vagy lejárt challenge\"}");
        return;
    };

    if (nowSeconds() - ch_info.created_at > AUTH_CHALLENGE_TTL) {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Lejárt challenge\"}");
        return;
    }
    if (ch_info.is_login) {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen challenge típus\"}");
        return;
    }

    const email = ch_info.email orelse {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Hiányzó e-mail a challenge-ből\"}");
        return;
    };
    const stored_origin = ch_info.origin;

    const cdj_bytes = base64UrlDecode(allocator, cdj_v.string) catch safeB64Decode(allocator, cdj_v.string) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen clientDataJSON kódolás\"}");
        return;
    };
    defer allocator.free(cdj_bytes);

    const cdj_parsed = std.json.parseFromSlice(std.json.Value, allocator, cdj_bytes, .{}) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen clientDataJSON\"}");
        return;
    };
    defer cdj_parsed.deinit();

    if (cdj_parsed.value == .object) {
        if (cdj_parsed.value.object.get("type")) |tv| {
            if (tv != .string or !std.mem.eql(u8, tv.string, "webauthn.create")) {
                try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen webauthn típus\"}");
                return;
            }
        }
        if (cdj_parsed.value.object.get("challenge")) |chv| {
            if (chv != .string or !std.mem.eql(u8, chv.string, challenge)) {
                try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Challenge nem egyezik\"}");
                return;
            }
        }
    }

    const ao_bytes = base64UrlDecode(allocator, ao_v.string) catch safeB64Decode(allocator, ao_v.string) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen attestationObject kódolás\"}");
        return;
    };
    defer allocator.free(ao_bytes);

    const ao = parseAttestationObject(allocator, ao_bytes) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Nem sikerült feldolgozni az attestationObject-et\"}");
        return;
    };
    defer allocator.free(ao.auth_data);

    var auth_data_result = parseAuthData(allocator, ao.auth_data) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen authData\"}");
        return;
    };
    defer auth_data_result.deinit();

    const pubkey_uncompressed = parseCoseP256Key(auth_data_result.cose_key) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Nem sikerült feldolgozni a nyilvános kulcsot\"}");
        return;
    };

    const rp_id = extractRpId(stored_origin);
    var expected_rp_id_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(rp_id, &expected_rp_id_hash, .{});

    if (!std.crypto.utils.timingSafeEql([32]u8, auth_data_result.rp_id_hash, expected_rp_id_hash)) {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"rpId nem egyezik\"}");
        return;
    }

    const cred_id_hex = bytesToHex(allocator, auth_data_result.cred_id) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
        return;
    };
    defer allocator.free(cred_id_hex);
    const pubkey_hex = bytesToHex(allocator, &pubkey_uncompressed) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
        return;
    };
    defer allocator.free(pubkey_hex);

    const token = generateAuthToken(allocator) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
        return;
    };
    defer allocator.free(token);

    {
        auth_mutex.lock();
        defer auth_mutex.unlock();

        if (auth_users.getPtr(email)) |user| {
            const gcid = global_allocator.dupe(u8, cred_id_hex) catch {
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            const gpk = global_allocator.dupe(u8, pubkey_hex) catch {
                global_allocator.free(gcid);
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            const grp = global_allocator.dupe(u8, rp_id) catch {
                global_allocator.free(gcid);
                global_allocator.free(gpk);
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            const pk = AuthPasskey{
                .credential_id = gcid,
                .public_key = gpk,
                .sign_count = auth_data_result.sign_count,
                .rp_id = grp,
            };
            user.passkeys.append(pk) catch {
                var pk_mut = pk;
                pk_mut.deinit(global_allocator);
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            addCredIdIndex(gcid, email);
        } else {
            const gid = global_allocator.dupe(u8, "") catch {
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            const gemail = global_allocator.dupe(u8, email) catch {
                global_allocator.free(gid);
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            const ghash = global_allocator.dupe(u8, "") catch {
                global_allocator.free(gid);
                global_allocator.free(gemail);
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            const gcid = global_allocator.dupe(u8, cred_id_hex) catch {
                global_allocator.free(gid);
                global_allocator.free(gemail);
                global_allocator.free(ghash);
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            const gpk = global_allocator.dupe(u8, pubkey_hex) catch {
                global_allocator.free(gid);
                global_allocator.free(gemail);
                global_allocator.free(ghash);
                global_allocator.free(gcid);
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            const grp = global_allocator.dupe(u8, rp_id) catch {
                global_allocator.free(gid);
                global_allocator.free(gemail);
                global_allocator.free(ghash);
                global_allocator.free(gcid);
                global_allocator.free(gpk);
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            const pk = AuthPasskey{
                .credential_id = gcid,
                .public_key = gpk,
                .sign_count = auth_data_result.sign_count,
                .rp_id = grp,
            };
            var new_user = AuthUser{
                .id = gid,
                .email = gemail,
                .password_hash = ghash,
                .passkeys = std.ArrayList(AuthPasskey).init(global_allocator),
                .created_at = nowSeconds(),
            };
            new_user.passkeys.append(pk) catch {
                var pk_mut = pk;
                pk_mut.deinit(global_allocator);
                new_user.passkeys.deinit();
                global_allocator.free(gid);
                global_allocator.free(gemail);
                global_allocator.free(ghash);
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            addCredIdIndex(gcid, email);
            const ukey = global_allocator.dupe(u8, email) catch {
                new_user.deinit(global_allocator);
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
            auth_users.put(ukey, new_user) catch {
                new_user.deinit(global_allocator);
                global_allocator.free(ukey);
                try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
                return;
            };
        }

        const tok_email = global_allocator.dupe(u8, email) catch {
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        const tok_info = AuthTokenInfo{ .email = tok_email, .expires_at = nowSeconds() + AUTH_TOKEN_TTL };
        const tok_key = global_allocator.dupe(u8, token) catch {
            var ti = tok_info;
            ti.deinit(global_allocator);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        auth_tokens.put(tok_key, tok_info) catch {
            var ti = tok_info;
            ti.deinit(global_allocator);
            global_allocator.free(tok_key);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
    }

    saveAuthToDisk();

    var resp_buf = std.ArrayList(u8).init(allocator);
    defer resp_buf.deinit();
    try resp_buf.writer().writeAll("{\"token\":");
    try writeJsonString(resp_buf.writer(), token);
    try resp_buf.writer().writeAll(",\"email\":");
    try writeJsonString(resp_buf.writer(), email);
    try resp_buf.writer().writeByte('}');
    try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, resp_buf.items);
}

fn handleWebAuthnLoginChallenge(allocator: std.mem.Allocator, writer: anytype, body: []const u8, headers: *const std.StringHashMap([]const u8)) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen kérés\"}");
        return;
    };
    defer parsed.deinit();

    var email: []const u8 = "";
    if (parsed.value == .object) {
        if (parsed.value.object.get("email")) |ev| if (ev == .string) { email = std.mem.trim(u8, ev.string, " \t\n\r"); };
    }

    const challenge = generateChallenge(allocator) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
        return;
    };
    defer allocator.free(challenge);

    const origin = getRequestOrigin(headers);

    {
        auth_mutex.lock();
        defer auth_mutex.unlock();
        pruneExpiredChallengesUnlocked();

        const ch_email: ?[]u8 = if (email.len > 0) global_allocator.dupe(u8, email) catch null else null;
        const ch_origin = global_allocator.dupe(u8, origin) catch {
            if (ch_email) |ce| global_allocator.free(ce);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        const ch_info = AuthChallengeInfo{
            .email = ch_email,
            .is_login = true,
            .created_at = nowSeconds(),
            .origin = ch_origin,
        };
        const ch_key = global_allocator.dupe(u8, challenge) catch {
            var ci = ch_info;
            ci.deinit(global_allocator);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        auth_challenges.put(ch_key, ch_info) catch {
            var ci = ch_info;
            ci.deinit(global_allocator);
            global_allocator.free(ch_key);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
    }

    const rp_id = extractRpId(origin);

    var resp_buf = std.ArrayList(u8).init(allocator);
    defer resp_buf.deinit();
    const w = resp_buf.writer();
    try w.writeByte('{');
    try w.writeAll("\"challenge\":");
    try writeJsonString(w, challenge);
    try w.writeAll(",\"rpId\":");
    try writeJsonString(w, rp_id);
    try w.writeByte('}');
    try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, resp_buf.items);
}

fn handleWebAuthnLoginVerify(allocator: std.mem.Allocator, writer: anytype, body: []const u8) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen kérés\"}");
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen kérés\"}"); return; }
    const obj = parsed.value.object;

    const challenge_v = obj.get("challenge") orelse { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Hiányzó challenge\"}"); return; };
    const cred_id_v = obj.get("credentialId") orelse { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Hiányzó credentialId\"}"); return; };
    const auth_data_v = obj.get("authenticatorData") orelse { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Hiányzó authenticatorData\"}"); return; };
    const cdj_v = obj.get("clientDataJSON") orelse { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Hiányzó clientDataJSON\"}"); return; };
    const sig_v = obj.get("signature") orelse { try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Hiányzó signature\"}"); return; };

    if (challenge_v != .string or cred_id_v != .string or auth_data_v != .string or cdj_v != .string or sig_v != .string) {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen mezők\"}");
        return;
    }

    const challenge = challenge_v.string;
    const cred_id_b64 = cred_id_v.string;

    var ch_fetched: ?struct { info: AuthChallengeInfo, key: []const u8 } = null;
    defer if (ch_fetched) |*cf| {
        cf.info.deinit(global_allocator);
        global_allocator.free(cf.key);
    };

    {
        auth_mutex.lock();
        defer auth_mutex.unlock();
        pruneExpiredChallengesUnlocked();
        if (auth_challenges.fetchRemove(challenge)) |kv| {
            ch_fetched = .{ .info = kv.value, .key = kv.key };
        }
    }

    const ch_info = if (ch_fetched) |*cf| &cf.info else {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen vagy lejárt challenge\"}");
        return;
    };

    if (nowSeconds() - ch_info.created_at > AUTH_CHALLENGE_TTL) {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Lejárt challenge\"}");
        return;
    }
    if (!ch_info.is_login) {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen challenge típus\"}");
        return;
    }

    const stored_origin = ch_info.origin;

    const cred_id_bytes = base64UrlDecode(allocator, cred_id_b64) catch safeB64Decode(allocator, cred_id_b64) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen credentialId kódolás\"}");
        return;
    };
    defer allocator.free(cred_id_bytes);

    const cred_id_hex_search = bytesToHex(allocator, cred_id_bytes) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
        return;
    };
    defer allocator.free(cred_id_hex_search);

    var found_email: []u8 = undefined;
    var found_pubkey_hex: []u8 = undefined;
    var found_sign_count: u32 = 0;
    var found = false;

    {
        cred_id_map_mutex.lock();
        const maybe_email = cred_id_to_email.get(cred_id_hex_search);
        cred_id_map_mutex.unlock();

        if (maybe_email) |idx_email| {
            auth_mutex.lock();
            defer auth_mutex.unlock();
            if (auth_users.getPtr(idx_email)) |user| {
                for (user.passkeys.items) |pk| {
                    if (std.mem.eql(u8, pk.credential_id, cred_id_hex_search)) {
                        found_email = global_allocator.dupe(u8, idx_email) catch break;
                        found_pubkey_hex = global_allocator.dupe(u8, pk.public_key) catch {
                            global_allocator.free(found_email);
                            break;
                        };
                        found_sign_count = pk.sign_count;
                        found = true;
                        break;
                    }
                }
            }
        }
    }

    if (!found) {
        auth_mutex.lock();
        defer auth_mutex.unlock();
        var it = auth_users.iterator();
        outer: while (it.next()) |entry| {
            for (entry.value_ptr.*.passkeys.items) |pk| {
                if (std.mem.eql(u8, pk.credential_id, cred_id_hex_search)) {
                    found_email = global_allocator.dupe(u8, entry.key_ptr.*) catch break :outer;
                    found_pubkey_hex = global_allocator.dupe(u8, pk.public_key) catch {
                        global_allocator.free(found_email);
                        break :outer;
                    };
                    found_sign_count = pk.sign_count;
                    found = true;
                    addCredIdIndex(cred_id_hex_search, entry.key_ptr.*);
                    break :outer;
                }
            }
        }
    }

    if (!found) {
        try sendHttpResponse(writer, 401, "Unauthorized", "application/json", &[_][2][]const u8{}, "{\"error\":\"Kulcs nem található\"}");
        return;
    }
    defer global_allocator.free(found_email);
    defer global_allocator.free(found_pubkey_hex);

    const pubkey_bytes = hexToBytes(allocator, found_pubkey_hex) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
        return;
    };
    defer allocator.free(pubkey_bytes);

    if (pubkey_bytes.len != 65) {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen kulcs méret\"}");
        return;
    }

    var pubkey_arr: [65]u8 = undefined;
    @memcpy(&pubkey_arr, pubkey_bytes[0..65]);

    const auth_data_bytes = base64UrlDecode(allocator, auth_data_v.string) catch safeB64Decode(allocator, auth_data_v.string) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen authenticatorData\"}");
        return;
    };
    defer allocator.free(auth_data_bytes);

    const cdj_bytes = base64UrlDecode(allocator, cdj_v.string) catch safeB64Decode(allocator, cdj_v.string) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen clientDataJSON kódolás\"}");
        return;
    };
    defer allocator.free(cdj_bytes);

    const cdj_parsed2 = std.json.parseFromSlice(std.json.Value, allocator, cdj_bytes, .{}) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen clientDataJSON\"}");
        return;
    };
    defer cdj_parsed2.deinit();

    if (cdj_parsed2.value == .object) {
        if (cdj_parsed2.value.object.get("type")) |tv| {
            if (tv != .string or !std.mem.eql(u8, tv.string, "webauthn.get")) {
                try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen webauthn típus\"}");
                return;
            }
        }
        if (cdj_parsed2.value.object.get("challenge")) |chv| {
            if (chv != .string or !std.mem.eql(u8, chv.string, challenge)) {
                try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Challenge nem egyezik\"}");
                return;
            }
        }
    }

    const rp_id = extractRpId(stored_origin);
    if (auth_data_bytes.len >= 37) {
        var expected_rp_hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(rp_id, &expected_rp_hash, .{});
        if (!std.crypto.utils.timingSafeEql([32]u8, auth_data_bytes[0..32].*, expected_rp_hash)) {
            try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"rpId nem egyezik\"}");
            return;
        }
    }

    const sig_bytes = base64UrlDecode(allocator, sig_v.string) catch safeB64Decode(allocator, sig_v.string) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen aláírás kódolás\"}");
        return;
    };
    defer allocator.free(sig_bytes);

    verifyWebAuthnSignature(sig_bytes, auth_data_bytes, cdj_bytes, pubkey_arr) catch {
        try sendHttpResponse(writer, 401, "Unauthorized", "application/json", &[_][2][]const u8{}, "{\"error\":\"Érvénytelen aláírás\"}");
        return;
    };

    if (auth_data_bytes.len >= 37) {
        const new_sign_count = std.mem.readInt(u32, auth_data_bytes[33..37], .big);
        if (new_sign_count > 0 and new_sign_count <= found_sign_count) {
            try sendHttpResponse(writer, 401, "Unauthorized", "application/json", &[_][2][]const u8{}, "{\"error\":\"Aláírási számláló visszalépés - lehetséges visszajátszás\"}");
            return;
        }
        if (new_sign_count > found_sign_count) {
            auth_mutex.lock();
            defer auth_mutex.unlock();
            if (auth_users.getPtr(found_email)) |user| {
                for (user.passkeys.items) |*pk| {
                    if (std.mem.eql(u8, pk.credential_id, cred_id_hex_search)) {
                        pk.sign_count = new_sign_count;
                        break;
                    }
                }
            }
        }
    }

    const token = generateAuthToken(allocator) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
        return;
    };
    defer allocator.free(token);

    {
        auth_mutex.lock();
        defer auth_mutex.unlock();
        const tok_email = global_allocator.dupe(u8, found_email) catch {
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        const tok_info = AuthTokenInfo{ .email = tok_email, .expires_at = nowSeconds() + AUTH_TOKEN_TTL };
        const tok_key = global_allocator.dupe(u8, token) catch {
            var ti = tok_info;
            ti.deinit(global_allocator);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
        auth_tokens.put(tok_key, tok_info) catch {
            var ti = tok_info;
            ti.deinit(global_allocator);
            global_allocator.free(tok_key);
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Szerverhiba\"}");
            return;
        };
    }

    saveAuthToDisk();

    var resp_buf = std.ArrayList(u8).init(allocator);
    defer resp_buf.deinit();
    try resp_buf.writer().writeAll("{\"token\":");
    try writeJsonString(resp_buf.writer(), token);
    try resp_buf.writer().writeAll(",\"email\":");
    try writeJsonString(resp_buf.writer(), found_email);
    try resp_buf.writer().writeByte('}');
    try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, resp_buf.items);
}

pub fn sendHttpResponse(writer: anytype, status: u16, status_text: []const u8, content_type: []const u8, extra_headers: []const [2][]const u8, body: []const u8) !void {
    try std.fmt.format(writer, "HTTP/1.1 {d} {s}\r\n", .{ status, status_text });
    try std.fmt.format(writer, "Content-Type: {s}\r\n", .{content_type});
    try std.fmt.format(writer, "Content-Length: {d}\r\n", .{body.len});
    try writer.writeAll("Connection: keep-alive\r\n");
    try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
    try writer.writeAll("Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n");
    try writer.writeAll("Access-Control-Allow-Headers: Content-Type, Authorization\r\n");
    for (extra_headers) |h| try std.fmt.format(writer, "{s}: {s}\r\n", .{ h[0], h[1] });
    try writer.writeAll("\r\n");
    try writer.writeAll(body);
}

fn sendSseHeaders(writer: anytype) !void {
    try writer.writeAll("HTTP/1.1 200 OK\r\n");
    try writer.writeAll("Content-Type: text/event-stream\r\n");
    try writer.writeAll("Cache-Control: no-cache\r\n");
    try writer.writeAll("X-Accel-Buffering: no\r\n");
    try writer.writeAll("Connection: keep-alive\r\n");
    try writer.writeAll("Access-Control-Allow-Origin: *\r\n");
    try writer.writeAll("Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n");
    try writer.writeAll("Access-Control-Allow-Headers: Content-Type, Authorization\r\n");
    try writer.writeAll("\r\n");
}

pub fn sendSseEvent(writer: anytype, data: []const u8) !void {
    try writer.writeAll("data: ");
    for (data) |c| {
        if (c == '\n') {
            try writer.writeAll("\ndata: ");
        } else if (c != '\r') {
            try writer.writeByte(c);
        }
    }
    try writer.writeAll("\n\n");
}

fn sendSsePing(writer: anytype) !void {
    try writer.writeAll(": ping\n\n");
}

fn urlDecode(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] == '%' and i + 2 < s.len) {
            const hex = s[i + 1 .. i + 3];
            const byte = std.fmt.parseInt(u8, hex, 16) catch {
                try result.append(s[i]);
                i += 1;
                continue;
            };
            try result.append(byte);
            i += 3;
        } else {
            try result.append(s[i]);
            i += 1;
        }
    }
    return result.toOwnedSlice();
}

fn urlEncode(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    for (s) |c| {
        if (std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.' or c == '~') {
            try result.append(c);
        } else {
            try std.fmt.format(result.writer(), "%{X:0>2}", .{c});
        }
    }
    return result.toOwnedSlice();
}

fn extractPreviewAndCount(msgs: []const Message) struct { preview: []const u8, count: usize } {
    var last_user: ?*const Message = null;
    for (msgs) |*msg| if (std.mem.eql(u8, msg.role, "user")) { last_user = msg; };

    var preview: []const u8 = "";
    if (last_user) |msg| {
        switch (msg.content) {
            .text => |t| preview = t,
            .parts => |parts| {
                for (parts.items) |part| {
                    if (std.mem.eql(u8, part.type, "text")) { if (part.text) |t| { preview = t; break; } }
                }
                if (preview.len == 0) {
                    for (parts.items) |part| {
                        if (std.mem.eql(u8, part.type, "image_url") or std.mem.eql(u8, part.type, "image_ref") or std.mem.eql(u8, part.type, "image_inline")) {
                            preview = "[kep]";
                            break;
                        }
                    }
                }
            },
        }
    }

    var count: usize = 0;
    for (msgs) |msg| {
        if (std.mem.eql(u8, msg.role, "user") or std.mem.eql(u8, msg.role, "assistant")) {
            if (std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
                const has_content = switch (msg.content) { .text => |t| t.len > 0, .parts => |p| p.items.len > 0 };
                if (!has_content) continue;
            }
            count += 1;
        }
    }

    if (preview.len > 80) {
        var end: usize = 80;
        while (end > 0 and (preview[end] & 0xC0) == 0x80) end -= 1;
        preview = preview[0..end];
    }
    return .{ .preview = preview, .count = count };
}

fn msgsDbToReadable(allocator: std.mem.Allocator, msgs: []const Message) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    const writer = buf.writer();

    try writer.writeByte('[');
    for (msgs, 0..) |msg, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.writeByte('{');
        try writer.writeAll("\"role\":");
        try writeJsonString(writer, msg.role);

        if (std.mem.eql(u8, msg.role, "tool")) {
            const content_str = switch (msg.content) {
                .text => |t| t,
                .parts => |parts| blk: {
                    for (parts.items) |part| {
                        if (std.mem.eql(u8, part.type, "text")) {
                            if (part.text) |t| break :blk t;
                        }
                    }
                    break :blk "";
                },
            };
            try writer.writeAll(",\"content\":");
            try writeJsonString(writer, content_str);
            if (msg.tool_call_id) |id| { try writer.writeAll(",\"tool_call_id\":"); try writeJsonString(writer, id); }
            try writer.writeAll(",\"is_tool_result\":true,\"has_image\":false,\"image_keys\":[]");
        } else {
            switch (msg.content) {
                .text => |t| {
                    try writer.writeAll(",\"content\":");
                    try writeJsonString(writer, t);
                    try writer.writeAll(",\"has_image\":false,\"image_keys\":[]");
                },
                .parts => |parts| {
                    var has_image = false;
                    var text_parts = std.ArrayList([]const u8).init(allocator);
                    defer text_parts.deinit();
                    var image_keys = std.ArrayList([]const u8).init(allocator);
                    defer image_keys.deinit();
                    for (parts.items) |part| {
                        const ptype = part.type;
                        if (std.mem.eql(u8, ptype, "image_ref") or std.mem.eql(u8, ptype, "image_url") or std.mem.eql(u8, ptype, "image_inline")) {
                            has_image = true;
                            if (std.mem.eql(u8, ptype, "image_ref") or std.mem.eql(u8, ptype, "image_url")) {
                                if (part.key) |k| try image_keys.append(k);
                            }
                        } else if (std.mem.eql(u8, ptype, "text")) {
                            if (part.text) |t| try text_parts.append(t);
                        }
                    }
                    var combined = std.ArrayList(u8).init(allocator);
                    defer combined.deinit();
                    for (text_parts.items, 0..) |t, ti| { if (ti > 0) try combined.append('\n'); try combined.appendSlice(t); }
                    try writer.writeAll(",\"content\":");
                    try writeJsonString(writer, combined.items);
                    try std.fmt.format(writer, ",\"has_image\":{}", .{has_image});
                    try writer.writeAll(",\"image_keys\":[");
                    for (image_keys.items, 0..) |k, ki| { if (ki > 0) try writer.writeByte(','); try writeJsonString(writer, k); }
                    try writer.writeByte(']');
                    try writer.writeAll(",\"parts\":[");
                    for (parts.items, 0..) |part, pi| { if (pi > 0) try writer.writeByte(','); try serializeContentPart(writer, part); }
                    try writer.writeByte(']');
                },
            }
            if (msg.tool_calls) |tcs| {
                try writer.writeAll(",\"tool_calls\":[");
                for (tcs.items, 0..) |tc, tci| {
                    if (tci > 0) try writer.writeByte(',');
                    try writer.writeAll("{\"id\":");
                    try writeJsonString(writer, tc.id);
                    try writer.writeAll(",\"type\":");
                    try writeJsonString(writer, tc.type);
                    try writer.writeAll(",\"function\":{\"name\":");
                    try writeJsonString(writer, tc.function.name);
                    try writer.writeAll(",\"arguments\":");
                    try writeJsonString(writer, tc.function.arguments);
                    try writer.writeAll("}}");
                }
                try writer.writeByte(']');
            }
        }
        try writer.writeByte('}');
    }
    try writer.writeByte(']');

    return buf.toOwnedSlice();
}

const StreamReaderType = std.net.Stream.Reader;
const BufferedReaderType = std.io.BufferedReader(4096, StreamReaderType);

const ConnectionContext = struct {
    conn: std.net.Server.Connection,
    allocator: std.mem.Allocator,
};

fn handleConnection(ctx: ConnectionContext) void {
    defer ctx.conn.stream.close();
    var buffered = std.io.bufferedReader(ctx.conn.stream.reader());
    while (true) {
        const keep_alive = handleRequest(ctx.allocator, ctx.conn, &buffered) catch break;
        if (!keep_alive) break;
    }
}

fn handleRequest(allocator: std.mem.Allocator, conn: std.net.Server.Connection, buffered: *BufferedReaderType) !bool {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const reader = buffered.reader();

    var first_line_buf = std.ArrayList(u8).init(arena_alloc);
    var line_len: usize = 0;
    while (true) {
        const byte = reader.readByte() catch return false;
        if (byte == '\n') break;
        if (byte != '\r') {
            if (line_len > 8192) return false;
            try first_line_buf.append(byte);
            line_len += 1;
        }
    }

    if (first_line_buf.items.len == 0) return false;

    var parts_iter = std.mem.splitScalar(u8, first_line_buf.items, ' ');
    const method_str = parts_iter.next() orelse return false;
    const path_full = parts_iter.next() orelse return false;

    const path_only = if (std.mem.indexOf(u8, path_full, "?")) |qi| path_full[0..qi] else path_full;

    var headers = std.StringHashMap([]const u8).init(arena_alloc);
    var content_length: usize = 0;
    var connection_close = false;

    while (true) {
        var header_line = std.ArrayList(u8).init(arena_alloc);
        var hline_len: usize = 0;
        while (true) {
            const byte = reader.readByte() catch return false;
            if (byte == '\n') break;
            if (byte != '\r') {
                if (hline_len > 8192) return false;
                try header_line.append(byte);
                hline_len += 1;
            }
        }
        if (header_line.items.len == 0) break;
        if (std.mem.indexOf(u8, header_line.items, ":")) |ci| {
            const hn = std.mem.trim(u8, header_line.items[0..ci], " \t");
            const hv = std.mem.trim(u8, header_line.items[ci + 1 ..], " \t");
            var lower_hn = try arena_alloc.alloc(u8, hn.len);
            for (hn, 0..) |c, idx| lower_hn[idx] = std.ascii.toLower(c);
            if (std.mem.eql(u8, lower_hn, "content-length")) {
                content_length = std.fmt.parseInt(usize, hv, 10) catch {
                    return error.InvalidContentLength;
                };
            }
            if (std.mem.eql(u8, lower_hn, "connection")) {
                if (std.ascii.eqlIgnoreCase(hv, "close")) connection_close = true;
            }
            try headers.put(lower_hn, try arena_alloc.dupe(u8, hv));
        }
    }

    const writer = conn.stream.writer();

    if (content_length > 50 * 1024 * 1024) {
        try conn.stream.writeAll("HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        return false;
    }

    var body: []u8 = &[_]u8{};
    if (content_length > 0) {
        body = try arena_alloc.alloc(u8, content_length);
        try reader.readNoEof(body);
    }

    const method = method_str;

    if (std.mem.eql(u8, method, "OPTIONS")) {
        const resp = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nConnection: keep-alive\r\nContent-Length: 0\r\n\r\n";
        conn.stream.writeAll(resp) catch {};
        return !connection_close;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path_only, "/")) {
        const f = std.fs.openFileAbsolute(index_file, .{}) catch {
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"index.html not found\"}");
            return !connection_close;
        };
        defer f.close();
        const content = f.readToEndAlloc(arena_alloc, 50 * 1024 * 1024) catch {
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Failed to read index.html\"}");
            return !connection_close;
        };
        try sendHttpResponse(writer, 200, "OK", "text/html; charset=utf-8", &[_][2][]const u8{.{ "Cache-Control", "no-cache, no-store, must-revalidate" }}, content);
        return !connection_close;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path_only, "/sw.js")) {
        const f = std.fs.openFileAbsolute(static_sw_file, .{}) catch {
            try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, "{\"detail\":\"Service worker not found\"}");
            return !connection_close;
        };
        defer f.close();
        const content = f.readToEndAlloc(arena_alloc, 10 * 1024 * 1024) catch {
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Failed to read sw.js\"}");
            return !connection_close;
        };
        try sendHttpResponse(writer, 200, "OK", "application/javascript", &[_][2][]const u8{.{ "Service-Worker-Allowed", "/" }}, content);
        return !connection_close;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path_only, "/health")) {
        const ai_ok = hpc_ai_api_key.len > 0;
        const status_str = if (ai_ok) "ok" else "degraded";
        const resp_body = try std.fmt.allocPrint(arena_alloc, "{{\"status\":\"{s}\",\"ai\":{}}}", .{ status_str, ai_ok });
        try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, resp_body);
        return !connection_close;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.startsWith(u8, path_only, "/static/")) {
        const rel = path_only[8..];
        const file_path = try std.fs.path.join(arena_alloc, &[_][]const u8{ static_dir, rel });
        const f = std.fs.openFileAbsolute(file_path, .{}) catch {
            try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, "{\"detail\":\"Not found\"}");
            return !connection_close;
        };
        defer f.close();
        const content = f.readToEndAlloc(arena_alloc, 50 * 1024 * 1024) catch {
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Failed to read file\"}");
            return !connection_close;
        };
        const ext = std.fs.path.extension(rel);
        const ct = if (std.mem.eql(u8, ext, ".js")) "application/javascript" else if (std.mem.eql(u8, ext, ".css")) "text/css" else if (std.mem.eql(u8, ext, ".json")) "application/json" else "application/octet-stream";
        try sendHttpResponse(writer, 200, "OK", ct, &[_][2][]const u8{}, content);
        return !connection_close;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.startsWith(u8, path_only, "/api/image/")) {
        const key_encoded = path_only[11..];
        const key = try urlDecode(arena_alloc, key_encoded);
        const image_dir = try std.fs.path.join(arena_alloc, &[_][]const u8{ project_root, "images" });
        const image_path = try std.fs.path.join(arena_alloc, &[_][]const u8{ image_dir, key });
        const f = std.fs.openFileAbsolute(image_path, .{}) catch {
            try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, "{\"detail\":\"Image not found\"}");
            return !connection_close;
        };
        defer f.close();
        const data = f.readToEndAlloc(arena_alloc, MAX_IMAGE_BYTES) catch {
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"error\":\"Failed to read image\"}");
            return !connection_close;
        };
        const detected = detectImageFormat(data);
        var mime = detected.mime;
        if (std.mem.eql(u8, mime, "application/octet-stream")) { const ext = std.fs.path.extension(key); if (extToMime(ext)) |m| mime = m; }
        try sendHttpResponse(writer, 200, "OK", mime, &[_][2][]const u8{.{ "Cache-Control", "public, max-age=86400" }}, data);
        return !connection_close;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path_only, "/api/sessions")) {
        const now = nowSeconds();

        var deleted_set = std.StringHashMap(void).init(arena_alloc);
        {
            deleted_sessions_mutex.lock();
            defer deleted_sessions_mutex.unlock();
            var it = deleted_sessions_map.iterator();
            while (it.next()) |entry| {
                if (now - entry.value_ptr.* <= SESSION_TTL) {
                    const key_copy = arena_alloc.dupe(u8, entry.key_ptr.*) catch continue;
                    deleted_set.put(key_copy, {}) catch {};
                }
            }
        }

        const SessionEntryOwned = struct { sid: []u8, updated_at: f64, preview: []u8, count: usize };
        var entries = std.ArrayList(SessionEntryOwned).init(arena_alloc);

        {
            sessions_mutex.lock();
            defer sessions_mutex.unlock();
            var it = sessions_map.iterator();
            while (it.next()) |entry| {
                if (deleted_set.get(entry.key_ptr.*) != null) continue;
                const pc = extractPreviewAndCount(entry.value_ptr.*.messages.items);
                const sid_copy = arena_alloc.dupe(u8, entry.key_ptr.*) catch continue;
                const preview_copy = arena_alloc.dupe(u8, pc.preview) catch continue;
                entries.append(.{
                    .sid = sid_copy,
                    .updated_at = entry.value_ptr.*.updated_at,
                    .preview = preview_copy,
                    .count = pc.count,
                }) catch {};
            }
        }

        std.sort.block(SessionEntryOwned, entries.items, {}, struct {
            fn lessThan(_: void, a: SessionEntryOwned, b: SessionEntryOwned) bool { return a.updated_at > b.updated_at; }
        }.lessThan);

        var result_buf = std.ArrayList(u8).init(arena_alloc);
        const rw = result_buf.writer();
        const limit = @min(entries.items.len, 100);
        try rw.writeAll("{\"sessions\":[");
        for (entries.items[0..limit], 0..) |entry, i| {
            if (i > 0) try rw.writeByte(',');
            try rw.writeByte('{');
            try rw.writeAll("\"session_id\":");
            try writeJsonString(rw, entry.sid);
            try rw.writeAll(",\"preview\":");
            try writeJsonString(rw, entry.preview);
            try std.fmt.format(rw, ",\"count\":{d}", .{entry.count});
            try std.fmt.format(rw, ",\"updated_at\":{d}", .{@as(u64, @intFromFloat(@max(0.0, entry.updated_at)))});
            try rw.writeByte('}');
        }
        try rw.writeAll("]}");

        try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, result_buf.items);
        return !connection_close;
    }

    if (std.mem.eql(u8, method, "GET") and std.mem.startsWith(u8, path_only, "/api/session/")) {
        const sid = path_only[13..];
        if (isDeleted(sid)) {
            try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, "{\"detail\":\"Session not found\"}");
            return !connection_close;
        }
        var readable: ?[]u8 = null;
        {
            sessions_mutex.lock();
            defer sessions_mutex.unlock();
            if (sessions_map.getPtr(sid)) |sess| {
                readable = msgsDbToReadable(arena_alloc, sess.messages.items) catch null;
            }
        }
        if (readable == null) {
            try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, "{\"detail\":\"Session not found\"}");
            return !connection_close;
        }
        var resp_buf = std.ArrayList(u8).init(arena_alloc);
        try resp_buf.writer().writeAll("{\"session_id\":");
        try writeJsonString(resp_buf.writer(), sid);
        try resp_buf.writer().writeAll(",\"messages\":");
        try resp_buf.writer().writeAll(readable.?);
        try resp_buf.writer().writeByte('}');
        try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, resp_buf.items);
        return !connection_close;
    }

    if (std.mem.eql(u8, method, "DELETE") and std.mem.startsWith(u8, path_only, "/api/session/")) {
        const sid = path_only[13..];
        markDeleted(sid);

        const session_lock = getOrCreateSessionLock(sid) catch {
            try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"detail\":\"Internal error\"}");
            return !connection_close;
        };
        session_lock.lock();

        var found = false;
        {
            sessions_mutex.lock();
            if (sessions_map.fetchRemove(sid)) |kv| {
                var sess = kv.value;
                sess.deinit(global_allocator);
                global_allocator.free(kv.key);
                found = true;
            }
            sessions_mutex.unlock();
        }

        session_lock.unlock();

        if (found) {
            saveSessionsToDisk();
            try sendHttpResponse(writer, 200, "OK", "application/json", &[_][2][]const u8{}, "{\"cleared\":true}");
        } else {
            unmarkDeleted(sid);
            try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, "{\"detail\":\"Session not found\"}");
        }
        return !connection_close;
    }

    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path_only, "/api/auth/register")) {
        try handleAuthRegister(arena_alloc, writer, body);
        return !connection_close;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path_only, "/api/auth/login")) {
        try handleAuthLogin(arena_alloc, writer, body);
        return !connection_close;
    }
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path_only, "/api/auth/me")) {
        try handleAuthMe(arena_alloc, writer, &headers);
        return !connection_close;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path_only, "/api/auth/logout")) {
        try handleAuthLogout(writer, &headers);
        return !connection_close;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path_only, "/api/auth/webauthn/register/challenge")) {
        try handleWebAuthnRegisterChallenge(arena_alloc, writer, body, &headers);
        return !connection_close;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path_only, "/api/auth/webauthn/register/verify")) {
        try handleWebAuthnRegisterVerify(arena_alloc, writer, body);
        return !connection_close;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path_only, "/api/auth/webauthn/login/challenge")) {
        try handleWebAuthnLoginChallenge(arena_alloc, writer, body, &headers);
        return !connection_close;
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path_only, "/api/auth/webauthn/login/verify")) {
        try handleWebAuthnLoginVerify(arena_alloc, writer, body);
        return !connection_close;
    }

    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path_only, "/api/chat")) {
        try handleChatEndpoint(arena_alloc, conn, body);
        return false;
    }

    try sendHttpResponse(writer, 404, "Not Found", "application/json", &[_][2][]const u8{}, "{\"detail\":\"Not found\"}");
    return !connection_close;
}

const ImagePayload = struct {
    mime: []u8,
    data: []u8,

    fn deinit(self: *ImagePayload, allocator: std.mem.Allocator) void {
        allocator.free(self.mime);
        allocator.free(self.data);
    }
};

const ChatRequestData = struct {
    session_id: ?[]u8,
    message: []u8,
    images: std.ArrayList(ImagePayload),

    fn deinit(self: *ChatRequestData, allocator: std.mem.Allocator) void {
        if (self.session_id) |s| allocator.free(s);
        allocator.free(self.message);
        for (self.images.items) |*img| img.deinit(allocator);
        self.images.deinit();
    }
};

fn parseChatRequest(allocator: std.mem.Allocator, body: []const u8) !ChatRequestData {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidRequest;
    const obj = parsed.value.object;

    const message_val = obj.get("message") orelse return error.MissingMessage;
    const message_src = if (message_val == .string) message_val.string else return error.MissingMessage;

    const trimmed = std.mem.trim(u8, message_src, " \t\n\r");
    if (trimmed.len == 0) return error.EmptyMessage;
    if (message_src.len > MAX_MESSAGE_LENGTH) return error.MessageTooLong;

    const message = try allocator.dupe(u8, message_src);
    errdefer allocator.free(message);

    const session_id: ?[]u8 = if (obj.get("session_id")) |sv| blk: {
        if (sv == .string and sv.string.len > 0) break :blk try allocator.dupe(u8, sv.string);
        break :blk null;
    } else null;
    errdefer if (session_id) |s| allocator.free(s);

    var images = std.ArrayList(ImagePayload).init(allocator);
    errdefer {
        for (images.items) |*img| img.deinit(allocator);
        images.deinit();
    }

    if (obj.get("images")) |iv| {
        if (iv == .array) {
            for (iv.array.items) |item| {
                if (item != .string) continue;
                const uri = item.string;
                if (!std.mem.startsWith(u8, uri, "data:")) continue;
                const comma_pos = std.mem.indexOf(u8, uri, ",") orelse continue;
                const header = uri[5..comma_pos];
                const b64_data = uri[comma_pos + 1 ..];
                const semi_pos = std.mem.indexOf(u8, header, ";");
                const mime_str = if (semi_pos) |pos| header[0..pos] else header;
                var valid_mime = false;
                for (ALLOWED_IMAGE_MIMES) |allowed| {
                    if (std.mem.eql(u8, mime_str, allowed)) { valid_mime = true; break; }
                }
                if (!valid_mime) continue;
                const mime_copy = try allocator.dupe(u8, mime_str);
                errdefer allocator.free(mime_copy);
                const data_copy = try allocator.dupe(u8, b64_data);
                errdefer allocator.free(data_copy);
                try images.append(ImagePayload{ .mime = mime_copy, .data = data_copy });
            }
        }
    }

    return ChatRequestData{ .session_id = session_id, .message = message, .images = images };
}

fn getOrCreateSession(allocator: std.mem.Allocator, session_id: ?[]const u8) !struct { sid: []u8, history: std.ArrayList(Message) } {
    var sid: []u8 = undefined;
    if (session_id) |given_sid| {
        sid = if (isDeleted(given_sid)) try generateUuid(allocator) else try allocator.dupe(u8, given_sid);
    } else {
        sid = try generateUuid(allocator);
    }
    errdefer allocator.free(sid);

    {
        sessions_mutex.lock();
        defer sessions_mutex.unlock();
        if (sessions_map.getPtr(sid)) |sess| {
            sess.updated_at = nowSeconds();
            var history = std.ArrayList(Message).init(allocator);
            errdefer {
                for (history.items) |*msg| msg.deinit(allocator);
                history.deinit();
            }
            for (sess.messages.items) |msg| try history.append(try msg.clone(allocator));
            return .{ .sid = sid, .history = history };
        }
    }

    evictSessions();

    sessions_mutex.lock();
    defer sessions_mutex.unlock();

    if (sessions_map.getPtr(sid)) |sess| {
        sess.updated_at = nowSeconds();
        var history = std.ArrayList(Message).init(allocator);
        errdefer {
            for (history.items) |*msg| msg.deinit(allocator);
            history.deinit();
        }
        for (sess.messages.items) |msg| try history.append(try msg.clone(allocator));
        return .{ .sid = sid, .history = history };
    }

    const now = nowSeconds();
    const key = try global_allocator.dupe(u8, sid);
    errdefer global_allocator.free(key);
    const sess = Session.init(global_allocator, now);
    try sessions_map.put(key, sess);

    const history = std.ArrayList(Message).init(allocator);
    return .{ .sid = sid, .history = history };
}

fn buildSseIdEvent(allocator: std.mem.Allocator, event_type: []const u8, field_name: []const u8, id_value: []const u8) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    const w = buf.writer();
    try w.writeByte('{');
    try w.writeAll("\"type\":");
    try writeJsonString(w, event_type);
    try w.writeByte(',');
    try writeJsonString(w, field_name);
    try w.writeByte(':');
    try writeJsonString(w, id_value);
    try w.writeByte('}');
    return buf.toOwnedSlice();
}

fn handleChatEndpoint(allocator: std.mem.Allocator, conn_in: std.net.Server.Connection, body: []const u8) !void {
    var conn = conn_in;
    const writer = conn.stream.writer();

    if (hpc_ai_api_key.len == 0) {
        try sendHttpResponse(writer, 503, "Service Unavailable", "application/json", &[_][2][]const u8{}, "{\"detail\":\"AI client not configured\"}");
        return;
    }

    var req_data = parseChatRequest(allocator, body) catch {
        try sendHttpResponse(writer, 400, "Bad Request", "application/json", &[_][2][]const u8{}, "{\"detail\":\"Invalid request\"}");
        return;
    };
    defer req_data.deinit(allocator);

    const session_result = getOrCreateSession(allocator, req_data.session_id) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"detail\":\"Session error\"}");
        return;
    };
    const sid = session_result.sid;
    defer allocator.free(sid);
    var history = session_result.history;
    defer { for (history.items) |*msg| msg.deinit(allocator); history.deinit(); }

    const session_lock = getOrCreateSessionLock(sid) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"detail\":\"Internal error\"}");
        return;
    };

    const msg_id = generateUuid(allocator) catch {
        try sendHttpResponse(writer, 500, "Internal Server Error", "application/json", &[_][2][]const u8{}, "{\"detail\":\"Internal error\"}");
        return;
    };
    defer allocator.free(msg_id);

    try sendSseHeaders(writer);

    {
        var sess_ev_buf = std.ArrayList(u8).init(allocator);
        defer sess_ev_buf.deinit();
        try sess_ev_buf.writer().writeAll("{\"type\":\"session\",\"session_id\":");
        try writeJsonString(sess_ev_buf.writer(), sid);
        try sess_ev_buf.writer().writeByte('}');
        try sendSseEvent(writer, sess_ev_buf.items);
    }

    session_lock.lock();
    defer session_lock.unlock();

    if (isDeleted(sid)) {
        try sendSseEvent(writer, "{\"type\":\"error\",\"message\":\"Session was deleted\"}");
        var done_buf = std.ArrayList(u8).init(allocator);
        defer done_buf.deinit();
        try done_buf.writer().writeAll("{\"type\":\"done\",\"session_id\":");
        try writeJsonString(done_buf.writer(), sid);
        try done_buf.writer().writeByte('}');
        try sendSseEvent(writer, done_buf.items);
        return;
    }

    {
        sessions_mutex.lock();
        defer sessions_mutex.unlock();
        if (!sessions_map.contains(sid)) {
            const key = try global_allocator.dupe(u8, sid);
            const now = nowSeconds();
            var sess = Session.init(global_allocator, now);
            for (history.items) |msg| try sess.messages.append(try msg.clone(global_allocator));
            try sessions_map.put(key, sess);
        }
        if (sessions_map.getPtr(sid)) |sess| {
            const role_copy = try global_allocator.dupe(u8, "user");
            const mid_copy = try global_allocator.dupe(u8, msg_id);
            var appended = false;
            errdefer if (!appended) global_allocator.free(role_copy);
            errdefer if (!appended) global_allocator.free(mid_copy);
            var user_content: MessageContent = blk_content: {
                if (req_data.images.items.len > 0) {
                    var parts = std.ArrayList(MessageContentPart).init(global_allocator);
                    errdefer if (!appended) {
                        for (parts.items) |*p| p.deinit(global_allocator);
                        parts.deinit();
                    };
                    if (req_data.message.len > 0) {
                        const text_type = try global_allocator.dupe(u8, "text");
                        errdefer if (!appended) global_allocator.free(text_type);
                        const text_val = try global_allocator.dupe(u8, req_data.message);
                        errdefer if (!appended) global_allocator.free(text_val);
                        try parts.append(MessageContentPart{
                            .type = text_type,
                            .text = text_val,
                            .media_type = null,
                            .data = null,
                            .key = null,
                            .image_url = null,
                        });
                    }
                    for (req_data.images.items) |img| {
                        const img_type = try global_allocator.dupe(u8, "image_url");
                        errdefer if (!appended) global_allocator.free(img_type);
                        const url_str = try std.fmt.allocPrint(global_allocator, "data:{s};base64,{s}", .{ img.mime, img.data });
                        errdefer if (!appended) global_allocator.free(url_str);
                        const detail_str = try global_allocator.dupe(u8, "auto");
                        errdefer if (!appended) global_allocator.free(detail_str);
                        try parts.append(MessageContentPart{
                            .type = img_type,
                            .text = null,
                            .media_type = null,
                            .data = null,
                            .key = null,
                            .image_url = MessageContentPart.ImageUrl{ .url = url_str, .detail = detail_str },
                        });
                    }
                    break :blk_content MessageContent{ .parts = parts };
                } else {
                    const content_copy = try global_allocator.dupe(u8, req_data.message);
                    break :blk_content MessageContent{ .text = content_copy };
                }
            };
            errdefer if (!appended) user_content.deinit(global_allocator);
            const user_msg = Message{
                .role = role_copy,
                .content = user_content,
                .tool_calls = null, .tool_call_id = null,
                .msg_id = mid_copy,
                .cached_size = null,
            };
            try sess.messages.append(user_msg);
            appended = true;
            try trimHistory(global_allocator, &sess.messages);
            sess.updated_at = nowSeconds();
        }
    }

    var iteration: usize = 0;
    while (iteration < MAX_TOOL_ITERATIONS) : (iteration += 1) {
        if (isDeleted(sid)) {
            try sendSseEvent(writer, "{\"type\":\"error\",\"message\":\"Session was deleted\"}");
            var done_buf = std.ArrayList(u8).init(allocator);
            defer done_buf.deinit();
            try done_buf.writer().writeAll("{\"type\":\"done\",\"session_id\":");
            try writeJsonString(done_buf.writer(), sid);
            try done_buf.writer().writeByte('}');
            try sendSseEvent(writer, done_buf.items);
            return;
        }

        var api_history: std.ArrayList(Message) = blk: {
            sessions_mutex.lock();
            defer sessions_mutex.unlock();
            if (sessions_map.getPtr(sid)) |sess| break :blk try buildApiMessages(allocator, sess.messages.items);
            break :blk std.ArrayList(Message).init(allocator);
        };
        defer { for (api_history.items) |*msg| msg.deinit(allocator); api_history.deinit(); }

        if (api_history.items.len == 0) {
            sendErrorEvent(allocator, writer, error.SessionDeleted) catch {};
            var done_buf = std.ArrayList(u8).init(allocator);
            defer done_buf.deinit();
            try done_buf.writer().writeAll("{\"type\":\"done\",\"session_id\":");
            try writeJsonString(done_buf.writer(), sid);
            try done_buf.writer().writeByte('}');
            try sendSseEvent(writer, done_buf.items);
            return;
        }

        const sys_prompt = try getSystemPrompt(allocator);
        defer allocator.free(sys_prompt);

        var all_messages = std.ArrayList(Message).init(allocator);
        defer { for (all_messages.items) |*msg| msg.deinit(allocator); all_messages.deinit(); }

        try all_messages.append(Message{
            .role = try allocator.dupe(u8, "system"),
            .content = MessageContent{ .text = try allocator.dupe(u8, sys_prompt) },
            .tool_calls = null, .tool_call_id = null, .msg_id = null, .cached_size = null,
        });
        for (api_history.items) |msg| try all_messages.append(try msg.clone(allocator));

        const max_tokens = safeMaxTokens(api_history.items);
        const request_body = try llm.buildOpenAIRequest(allocator, "zai-org/glm-5.1", all_messages.items, max_tokens, true, true);
        defer allocator.free(request_body);

        var tool_call_chunks = std.AutoHashMap(usize, *llm.ToolCallAccumulator).init(allocator);
        defer { var it = tool_call_chunks.iterator(); while (it.next()) |entry| entry.value_ptr.*.deinit(); tool_call_chunks.deinit(); }

        var finish_reason: ?[]u8 = null;
        defer if (finish_reason) |fr| allocator.free(fr);
        var turn_content = std.ArrayList(u8).init(allocator);
        defer turn_content.deinit();
        var reasoning_started = false;
        var reasoning_done = false;

        var sse_writer = llm.SseWriter{ .conn = &conn, .allocator = allocator };
        var ctx = llm.ChatStreamContext{
            .sse = &sse_writer,
            .tool_call_chunks = &tool_call_chunks,
            .finish_reason = &finish_reason,
            .turn_content = &turn_content,
            .reasoning_started = &reasoning_started,
            .reasoning_done = &reasoning_done,
            .allocator = allocator,
        };

        const base_url = "https://api.hpc-ai.com/inference/v1";
        llm.callOpenAIStreaming(allocator, base_url, hpc_ai_api_key, request_body, llm.chatStreamCallback, &ctx) catch |err| {
            std.log.err("chat stream failed: {s}", .{@errorName(err)});
            rollbackSessionTurn(sid, msg_id);
            sendErrorEvent(allocator, writer, err) catch {};
            var done_buf = std.ArrayList(u8).init(allocator);
            defer done_buf.deinit();
            try done_buf.writer().writeAll("{\"type\":\"done\",\"session_id\":");
            try writeJsonString(done_buf.writer(), sid);
            try done_buf.writer().writeByte('}');
            try sendSseEvent(writer, done_buf.items);
            return;
        };

        if (reasoning_started and !reasoning_done) try sendSseEvent(writer, "{\"type\":\"reasoning_end\"}");

        {
            var tcc_it = tool_call_chunks.iterator();
            while (tcc_it.next()) |entry| {
                const acc = entry.value_ptr.*;
                const full_args = try acc.getArguments();
                for (acc.arg_parts.items) |part| allocator.free(part);
                acc.arg_parts.clearRetainingCapacity();
                acc.arg_parts.append(full_args) catch {
                    allocator.free(full_args);
                };
            }
        }

        const is_tool_call = if (finish_reason) |fr| std.mem.eql(u8, fr, "tool_calls") else false;

        if (is_tool_call and tool_call_chunks.count() > 0) {
            var sorted_indices = std.ArrayList(usize).init(allocator);
            defer sorted_indices.deinit();
            var it = tool_call_chunks.keyIterator();
            while (it.next()) |k| try sorted_indices.append(k.*);
            std.sort.block(usize, sorted_indices.items, {}, std.sort.asc(usize));

            var tool_calls_list = std.ArrayList(ToolCall).init(allocator);
            defer { for (tool_calls_list.items) |*tc| tc.deinit(allocator); tool_calls_list.deinit(); }

            for (sorted_indices.items) |idx| {
                const acc = tool_call_chunks.get(idx).?;
                const resolved_id = if (acc.id.len > 0) try allocator.dupe(u8, acc.id) else blk2: {
                    const short = try generateShortHex(allocator);
                    defer allocator.free(short);
                    break :blk2 try std.fmt.allocPrint(allocator, "tc_{s}", .{short});
                };
                errdefer allocator.free(resolved_id);

                const raw_args = if (acc.arg_parts.items.len > 0) acc.arg_parts.items[0] else "";
                const args_str = blk_args: {
                    if (std.json.parseFromSlice(std.json.Value, allocator, raw_args, .{})) |parsed_check| {
                        parsed_check.deinit();
                        break :blk_args try allocator.dupe(u8, raw_args);
                    } else |_| {
                        var raw_escaped = std.ArrayList(u8).init(allocator);
                        defer raw_escaped.deinit();
                        try writeJsonString(raw_escaped.writer(), raw_args);
                        break :blk_args try std.fmt.allocPrint(allocator, "{{\"raw_invalid_arguments\":{s}}}", .{raw_escaped.items});
                    }
                };
                try tool_calls_list.append(ToolCall{
                    .id = resolved_id,
                    .type = try allocator.dupe(u8, "function"),
                    .function = .{ .name = try allocator.dupe(u8, acc.name), .arguments = args_str },
                });
            }

            {
                sessions_mutex.lock();
                defer sessions_mutex.unlock();
                if (sessions_map.getPtr(sid)) |sess| {
                    if (!isDeleted(sid)) {
                        var tcs_clone = std.ArrayList(ToolCall).init(global_allocator);
                        var tcs_appended = false;
                        errdefer if (!tcs_appended) {
                            for (tcs_clone.items) |*tc| tc.deinit(global_allocator);
                            tcs_clone.deinit();
                        };
                        for (tool_calls_list.items) |tc| try tcs_clone.append(try tc.clone(global_allocator));
                        const role_copy = try global_allocator.dupe(u8, "assistant");
                        errdefer if (!tcs_appended) global_allocator.free(role_copy);
                        const content_copy = try global_allocator.dupe(u8, turn_content.items);
                        errdefer if (!tcs_appended) global_allocator.free(content_copy);
                        const asst_msg = Message{
                            .role = role_copy,
                            .content = MessageContent{ .text = content_copy },
                            .tool_calls = tcs_clone, .tool_call_id = null, .msg_id = null, .cached_size = null,
                        };
                        try sess.messages.append(asst_msg);
                        tcs_appended = true;
                        try trimHistory(global_allocator, &sess.messages);
                        sess.updated_at = nowSeconds();
                    }
                }
            }

            var has_search_tool = false;
            for (tool_calls_list.items) |tc| {
                if (std.mem.eql(u8, tc.function.name, "exa_search")) {
                    has_search_tool = true;
                    {
                        var ss_buf = std.ArrayList(u8).init(allocator);
                        defer ss_buf.deinit();
                        const ssw = ss_buf.writer();
                        var first_query_owned: ?[]u8 = null;
                        defer if (first_query_owned) |fq| allocator.free(fq);
                        const first_query: []const u8 = blk_q: {
                            if (std.json.parseFromSlice(std.json.Value, allocator, tc.function.arguments, .{})) |pa| {
                                defer pa.deinit();
                                if (pa.value == .object) {
                                    if (pa.value.object.get("queries")) |qv| {
                                        if (qv == .array and qv.array.items.len > 0 and qv.array.items[0] == .string) {
                                            first_query_owned = allocator.dupe(u8, qv.array.items[0].string) catch null;
                                            if (first_query_owned) |fq| break :blk_q fq;
                                        }
                                    }
                                }
                            } else |_| {}
                            break :blk_q "Web keresés";
                        };
                        try ssw.writeAll("{\"type\":\"search_start\",\"id\":");
                        try writeJsonString(ssw, tc.id);
                        try ssw.writeAll(",\"query\":");
                        try writeJsonString(ssw, first_query);
                        try ssw.writeByte('}');
                        try sendSseEvent(writer, ss_buf.items);
                    }
                    const tool_msg = exa.executeExaToolCall(allocator, &sse_writer, tc.id, tc.function.arguments, sid) catch |err| blk3: {
                        std.log.err("exa search failed: {s}", .{@errorName(err)});
                        var ev_buf = std.ArrayList(u8).init(allocator);
                        defer ev_buf.deinit();
                        const ew = ev_buf.writer();
                        try ew.writeAll("{\"type\":\"search_error\",\"id\":");
                        try writeJsonString(ew, tc.id);
                        try ew.writeAll(",\"error\":");
                        const friendly_err = errorToHungarian(err);
                        if (friendly_err.len > 0) { try writeJsonString(ew, friendly_err); } else { try writeJsonString(ew, @errorName(err)); }
                        try ew.writeByte('}');
                        try sendSseEvent(writer, ev_buf.items);
                        const err_text = try std.fmt.allocPrint(allocator, "Search failed: {s}", .{@errorName(err)});
                        errdefer allocator.free(err_text);
                        const err_role = try allocator.dupe(u8, "tool");
                        errdefer allocator.free(err_role);
                        const err_tcid = try allocator.dupe(u8, tc.id);
                        break :blk3 Message{
                            .role = err_role,
                            .content = MessageContent{ .text = err_text },
                            .tool_calls = null, .tool_call_id = err_tcid,
                            .msg_id = null, .cached_size = null,
                        };
                    };
                    var tool_msg_local = tool_msg;
                    defer tool_msg_local.deinit(allocator);
                    {
                        sessions_mutex.lock();
                        defer sessions_mutex.unlock();
                        if (sessions_map.getPtr(sid)) |sess| {
                            if (!isDeleted(sid)) {
                                try sess.messages.append(try tool_msg_local.clone(global_allocator));
                                try trimHistory(global_allocator, &sess.messages);
                                sess.updated_at = nowSeconds();
                            }
                        }
                    }
                } else {
                    var ev_buf = std.ArrayList(u8).init(allocator);
                    defer ev_buf.deinit();
                    const ew = ev_buf.writer();
                    try ew.writeAll("{\"type\":\"tool_error\",\"id\":");
                    try writeJsonString(ew, tc.id);
                    try ew.writeAll(",\"error\":");
                    const err_msg = try std.fmt.allocPrint(allocator, "Unknown tool: {s}", .{tc.function.name});
                    defer allocator.free(err_msg);
                    try writeJsonString(ew, err_msg);
                    try ew.writeByte('}');
                    try sendSseEvent(writer, ev_buf.items);

                    const t_role = global_allocator.dupe(u8, "tool") catch continue;
                    const t_content = std.fmt.allocPrint(global_allocator, "Error: unknown tool '{s}'", .{tc.function.name}) catch { global_allocator.free(t_role); continue; };
                    const t_tcid = global_allocator.dupe(u8, tc.id) catch { global_allocator.free(t_role); global_allocator.free(t_content); continue; };
                    const tool_msg = Message{
                        .role = t_role, .content = MessageContent{ .text = t_content },
                        .tool_calls = null, .tool_call_id = t_tcid,
                        .msg_id = null, .cached_size = null,
                    };
                    sessions_mutex.lock();
                    defer sessions_mutex.unlock();
                    if (sessions_map.getPtr(sid)) |sess| {
                        if (!isDeleted(sid)) {
                            sess.messages.append(tool_msg) catch { var tm = tool_msg; tm.deinit(global_allocator); };
                            try trimHistory(global_allocator, &sess.messages);
                            sess.updated_at = nowSeconds();
                        } else {
                            var tm = tool_msg;
                            tm.deinit(global_allocator);
                        }
                    } else {
                        var tm = tool_msg;
                        tm.deinit(global_allocator);
                    }
                }
            }

            if (has_search_tool) try sendSseEvent(writer, "{\"type\":\"search_done\"}");
        } else {
            var final_content = turn_content.items;
            var extra_content: ?[]u8 = null;
            defer if (extra_content) |ec| allocator.free(ec);

            if (finish_reason != null and std.mem.eql(u8, finish_reason.?, "length") and final_content.len > 0) {
                const trunc_msg = "\n\n[Response truncated due to length limit]";
                extra_content = try std.fmt.allocPrint(allocator, "{s}{s}", .{ final_content, trunc_msg });
                final_content = extra_content.?;
                try sendSseEvent(writer, "{\"type\":\"content\",\"text\":\"\\n\\n[Response truncated due to length limit]\"}");
            }

            {
                sessions_mutex.lock();
                defer sessions_mutex.unlock();
                if (sessions_map.getPtr(sid)) |sess| {
                    if (!isDeleted(sid)) {
                        if (final_content.len > 0) {
                            const r = try global_allocator.dupe(u8, "assistant");
                            errdefer global_allocator.free(r);
                            const c = try global_allocator.dupe(u8, final_content);
                            const asst_msg = Message{
                                .role = r, .content = MessageContent{ .text = c },
                                .tool_calls = null, .tool_call_id = null, .msg_id = null, .cached_size = null,
                            };
                            try sess.messages.append(asst_msg);
                        }
                        try trimHistory(global_allocator, &sess.messages);
                        sess.updated_at = nowSeconds();
                    }
                }
            }
            saveSessionsToDisk();

            var done_buf = std.ArrayList(u8).init(allocator);
            defer done_buf.deinit();
            try done_buf.writer().writeAll("{\"type\":\"done\",\"session_id\":");
            try writeJsonString(done_buf.writer(), sid);
            try done_buf.writer().writeByte('}');
            try sendSseEvent(writer, done_buf.items);
            return;
        }
    }

    {
        sessions_mutex.lock();
        defer sessions_mutex.unlock();
        if (sessions_map.getPtr(sid)) |sess| {
            if (!isDeleted(sid)) {
                const r = try global_allocator.dupe(u8, "assistant");
                errdefer global_allocator.free(r);
                const c = try global_allocator.dupe(u8, "Tool iteration limit reached. Please try your question again.");
                const asst_msg = Message{
                    .role = r, .content = MessageContent{ .text = c },
                    .tool_calls = null, .tool_call_id = null, .msg_id = null, .cached_size = null,
                };
                try sess.messages.append(asst_msg);
                try trimHistory(global_allocator, &sess.messages);
                sess.updated_at = nowSeconds();
            }
        }
    }
    saveSessionsToDisk();

    try sendSseEvent(writer, "{\"type\":\"content\",\"text\":\"Tool iteration limit reached. Please try your question again.\"}");
    var done_buf2 = std.ArrayList(u8).init(allocator);
    defer done_buf2.deinit();
    try done_buf2.writer().writeAll("{\"type\":\"done\",\"session_id\":");
    try writeJsonString(done_buf2.writer(), sid);
    try done_buf2.writer().writeByte('}');
    try sendSseEvent(writer, done_buf2.items);
}

pub fn main() !void {
    global_allocator = gpa.allocator();
    defer _ = gpa.deinit();

    hpc_ai_api_key = try getEnvAlloc(global_allocator, "HPC_AI_API_KEY", "");
    exa_api_key = try getEnvAlloc(global_allocator, "EXA_API_KEY", "");
    server_host = try getEnvAlloc(global_allocator, "HOST", "0.0.0.0");

    const port_str = try getEnvAlloc(global_allocator, "PORT", "5000");
    defer global_allocator.free(port_str);
    server_port = std.fmt.parseInt(u16, port_str, 10) catch 5000;

    const project_root_env = try getEnvAlloc(global_allocator, "PROJECT_ROOT", "");
    if (project_root_env.len > 0) {
        project_root = project_root_env;
    } else {
        global_allocator.free(project_root_env);
        const cwd_path = std.fs.cwd().realpathAlloc(global_allocator, ".") catch blk: {
            const exe_dir = try std.fs.selfExeDirPathAlloc(global_allocator);
            break :blk exe_dir;
        };
        project_root = cwd_path;
    }

    index_file = try std.fs.path.join(global_allocator, &[_][]const u8{ project_root, "index.html" });
    static_dir = try std.fs.path.join(global_allocator, &[_][]const u8{ project_root, "static" });
    static_sw_file = try std.fs.path.join(global_allocator, &[_][]const u8{ static_dir, "sw.js" });
    persistence_file = try std.fs.path.join(global_allocator, &[_][]const u8{ project_root, "sessions_db.json" });
    auth_file = try std.fs.path.join(global_allocator, &[_][]const u8{ project_root, "auth_db.json" });

    sessions_map = std.StringHashMap(Session).init(global_allocator);
    session_locks_map = std.StringHashMap(*std.Thread.Mutex).init(global_allocator);
    deleted_sessions_map = std.StringHashMap(f64).init(global_allocator);
    session_exa_limiters_map = std.StringHashMap(*RateLimiter).init(global_allocator);
    exa_limiter = RateLimiter.init(global_allocator, EXA_QPS_LIMIT);
    auth_users = std.StringHashMap(AuthUser).init(global_allocator);
    auth_tokens = std.StringHashMap(AuthTokenInfo).init(global_allocator);
    auth_challenges = std.StringHashMap(AuthChallengeInfo).init(global_allocator);
    cred_id_to_email = std.StringHashMap([]u8).init(global_allocator);

    loadSessionsFromDisk();
    loadAuthFromDisk();

    std.log.info("Starting Helium server on {s}:{d}", .{ server_host, server_port });

    const addr = std.net.Address.parseIp(server_host, server_port) catch blk: {
        const list = try std.net.getAddressList(global_allocator, server_host, server_port);
        defer list.deinit();
        if (list.addrs.len == 0) return error.HostNotFound;
        break :blk list.addrs[0];
    };
    var server = try addr.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.log.info("Helium server running on port {d}", .{server_port});

    while (true) {
        const conn = server.accept() catch |err| { std.log.err("Accept error: {}", .{err}); continue; };
        const thread_alloc = global_allocator;
        const ctx = ConnectionContext{ .conn = conn, .allocator = thread_alloc };
        const thread = std.Thread.spawn(.{}, handleConnection, .{ctx}) catch |err| {
            std.log.err("Failed to spawn thread: {}", .{err});
            conn.stream.close();
            continue;
        };
        thread.detach();
    }
}
