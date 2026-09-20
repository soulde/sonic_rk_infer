#include "rknn_api_compat.h"

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelSubscriber;

constexpr int kDof = 29;
constexpr int kEncoderInput = 1247;
constexpr int kDecoderInput = 994;
constexpr int kToken = 64;
constexpr int kAction = 29;
constexpr double kControlHz = 50.0;
constexpr double kPublishHz = 500.0;

const std::array<int, kDof> kIsaacToMujoco = {
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28};
const std::array<int, kDof> kMujocoToIsaac = [] {
  std::array<int, kDof> result{};
  for (int i = 0; i < kDof; ++i) result[kIsaacToMujoco[i]] = i;
  return result;
}();

const std::array<float, kDof> kDefaultAngles = {
    -0.312f, 0.f, 0.f, 0.669f, -0.363f, 0.f, -0.312f, 0.f, 0.f, 0.669f,
    -0.363f, 0.f, 0.f, 0.f, 0.f, 0.2f, 0.2f, 0.f, 0.6f, 0.f, 0.f, 0.f,
    0.2f, -0.2f, 0.f, 0.6f, 0.f, 0.f, 0.f};
const std::array<float, kDof> kArmature = {
    0.025101925f, 0.025101925f, 0.010177520f, 0.025101925f, 0.003609725f,
    0.003609725f, 0.025101925f, 0.025101925f, 0.010177520f, 0.025101925f,
    0.003609725f, 0.003609725f, 0.010177520f, 0.003609725f, 0.003609725f,
    0.003609725f, 0.003609725f, 0.003609725f, 0.003609725f, 0.003609725f,
    0.00425f, 0.00425f, 0.003609725f, 0.003609725f, 0.003609725f,
    0.003609725f, 0.003609725f, 0.00425f, 0.00425f};
const std::array<float, kDof> kEffort = {
    139.f, 139.f, 88.f, 139.f, 25.f, 25.f, 139.f, 139.f, 88.f, 139.f,
    25.f, 25.f, 88.f, 25.f, 25.f, 25.f, 25.f, 25.f, 25.f, 25.f, 5.f,
    5.f, 25.f, 25.f, 25.f, 25.f, 25.f, 5.f, 5.f};

std::array<float, kDof> make_kp() {
  std::array<float, kDof> out{};
  const float omega = 20.f * static_cast<float>(M_PI);
  for (int i = 0; i < kDof; ++i) out[i] = kArmature[i] * omega * omega;
  for (int i : {4, 5, 10, 11, 13, 14}) out[i] *= 2.f;
  return out;
}
std::array<float, kDof> make_kd() {
  std::array<float, kDof> out{};
  const float omega = 20.f * static_cast<float>(M_PI);
  for (int i = 0; i < kDof; ++i) out[i] = 4.f * kArmature[i] * omega;
  for (int i : {4, 5, 10, 11, 13, 14}) out[i] *= 2.f;
  return out;
}
std::array<float, kDof> make_scale() {
  std::array<float, kDof> out{};
  const float omega = 20.f * static_cast<float>(M_PI);
  for (int i = 0; i < kDof; ++i) out[i] = .25f * kEffort[i] / (kArmature[i] * omega * omega);
  return out;
}
const auto kKp = make_kp();
const auto kKd = make_kd();
const auto kActionScale = make_scale();

uint32_t crc32(uint32_t* ptr, uint32_t len) {
  uint32_t crc = 0xffffffffu;
  constexpr uint32_t poly = 0x04c11db7u;
  for (uint32_t i = 0; i < len; ++i) {
    uint32_t data = ptr[i];
    for (uint32_t bit = 0; bit < 32; ++bit) {
      const bool high = (crc & 0x80000000u) != 0;
      crc <<= 1;
      if (high) crc ^= poly;
      if (data & (1u << (31 - bit))) crc ^= poly;
    }
  }
  return crc;
}

class RknnModel {
 public:
  RknnModel(std::string name, std::string path, size_t input, size_t output)
      : name_(std::move(name)), path_(std::move(path)), input_size_(input), output_size_(output) {}
  ~RknnModel() { if (ctx_) rknn_destroy(ctx_); }
  void open() {
    std::ifstream file(path_, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open " + path_);
    const auto size = file.tellg();
    model_.resize(static_cast<size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(model_.data()), size);
    check(rknn_init(&ctx_, model_.data(), static_cast<uint32_t>(model_.size()), 0, nullptr), "rknn_init");
    rknn_input_output_num io{};
    check(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "rknn_query io");
    if (io.n_input != 1 || io.n_output != 1) throw std::runtime_error(name_ + " requires one input/output");
    input_attr_.index = output_attr_.index = 0;
    check(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_)), "input attr");
    check(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attr_, sizeof(output_attr_)), "output attr");
    if (input_attr_.n_elems != input_size_ || output_attr_.n_elems != output_size_)
      throw std::runtime_error(name_ + " tensor shape mismatch");
    std::cout << name_ << " input=" << input_attr_.n_elems << " output=" << output_attr_.n_elems
              << " input_type=" << input_attr_.type << " output_type=" << output_attr_.type << '\n';
  }
  std::vector<float> run(const std::vector<float>& input) {
    if (input.size() != input_size_) throw std::runtime_error(name_ + " bad input size");
    rknn_input in{};
    in.index = 0; in.buf = const_cast<float*>(input.data());
    in.size = static_cast<uint32_t>(input.size() * sizeof(float));
    in.type = RKNN_TENSOR_FLOAT32; in.fmt = RKNN_TENSOR_UNDEFINED;
    check(rknn_inputs_set(ctx_, 1, &in), name_ + " inputs_set");
    check(rknn_run(ctx_, nullptr), name_ + " run");
    rknn_output out{}; out.index = 0; out.want_float = 1;
    check(rknn_outputs_get(ctx_, 1, &out, nullptr), name_ + " outputs_get");
    if (out.size < output_size_ * sizeof(float)) {
      rknn_outputs_release(ctx_, 1, &out);
      throw std::runtime_error(name_ + " output size mismatch");
    }
    std::vector<float> result(output_size_);
    std::memcpy(result.data(), out.buf, output_size_ * sizeof(float));
    check(rknn_outputs_release(ctx_, 1, &out), name_ + " outputs_release");
    return result;
  }
 private:
  static void check(int ret, const std::string& what) {
    if (ret) throw std::runtime_error(what + " failed: " + std::to_string(ret));
  }
  std::string name_, path_;
  size_t input_size_, output_size_;
  rknn_context ctx_ = 0;
  std::vector<uint8_t> model_;
  rknn_tensor_attr input_attr_{}, output_attr_{};
};

struct Snapshot {
  LowState state;
  Clock::time_point received;
};
class StateBuffer {
 public:
  void update(const LowState& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_ = std::make_shared<Snapshot>(Snapshot{value, Clock::now()});
  }
  std::shared_ptr<const Snapshot> latest() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
  }
 private:
  mutable std::mutex mutex_;
  std::shared_ptr<Snapshot> latest_;
};

struct Command {
  std::array<float, kDof> q{}, dq{}, kp{}, kd{}, tau{};
};

struct Quat { float w, x, y, z; };
Quat qmul(Quat a, Quat b) { return {a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z, a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y, a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x, a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w}; }
Quat qconj(Quat q) { return {q.w, -q.x, -q.y, -q.z}; }
Quat qnorm(Quat q) { const float n=std::sqrt(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z); return {q.w/n,q.x/n,q.y/n,q.z/n}; }
Quat heading(Quat q) { q=qnorm(q); const float yaw=std::atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)); return {std::cos(yaw/2),0,0,std::sin(yaw/2)}; }
std::array<float, 6> rot6(Quat q) { q=qnorm(q); return {1-2*(q.y*q.y+q.z*q.z),2*(q.x*q.y-q.z*q.w),2*(q.x*q.y+q.z*q.w),1-2*(q.x*q.x+q.z*q.z),2*(q.x*q.z-q.y*q.w),2*(q.y*q.z+q.x*q.w)}; }
std::array<float, 3> gravity(Quat q) { return {-2*(q.x*q.z-q.w*q.y),-2*(q.w*q.x+q.y*q.z),-(1-2*(q.x*q.x+q.y*q.y))}; }

std::vector<float> csv_row(const std::string& path, int expected) {
  std::ifstream file(path); if (!file) throw std::runtime_error("cannot open " + path);
  std::string line; std::getline(file, line); std::vector<float> row; row.reserve(expected);
  while (std::getline(file,line)) { std::stringstream ss(line); std::string cell; row.clear(); while(std::getline(ss,cell,',')) row.push_back(std::stof(cell)); if ((int)row.size()==expected) return row; }
  throw std::runtime_error("empty csv " + path);
}
struct Motion {
  std::vector<std::array<float,kDof>> pos, vel; std::vector<Quat> root;
  explicit Motion(const std::string& dir) {
    std::ifstream fp(dir+"/joint_pos.csv"), fv(dir+"/joint_vel.csv"), fq(dir+"/body_quat.csv");
    if(!fp||!fv||!fq) throw std::runtime_error("motion csv files missing");
    std::string hp,hv,hq; std::getline(fp,hp);std::getline(fv,hv);std::getline(fq,hq);
    std::string lp,lv,lq;
    while(std::getline(fp,lp)&&std::getline(fv,lv)&&std::getline(fq,lq)) {
      std::array<float,kDof> p{},v{}; std::stringstream sp(lp),sv(lv); std::string c; std::vector<float> vp,vv,vq;
      while(std::getline(sp,c,',')) vp.push_back(std::stof(c)); while(std::getline(sv,c,',')) vv.push_back(std::stof(c));
      std::stringstream sq(lq); while(std::getline(sq,c,',')) vq.push_back(std::stof(c));
      if(vp.size()!=kDof||vv.size()!=kDof||vq.size()<4) continue;
      // The official CSV is already in IsaacLab joint order.  Keep that
      // order for the policy input; only the DDS motor command is remapped.
      for(int isaac=0;isaac<kDof;++isaac){p[isaac]=vp[isaac];v[isaac]=vv[isaac];}
      pos.push_back(p);vel.push_back(v);root.push_back(qnorm({vq[0],vq[1],vq[2],vq[3]}));
    }
    if(pos.empty()) throw std::runtime_error("motion has no frames");
  }
  int frames() const { return static_cast<int>(pos.size()); }
};

std::array<float,93> state_frame(const LowState& state, const std::array<float,kDof>& last_action) {
  std::array<float,93> out{};
  for(int i=0;i<3;++i) out[i]=state.imu_state().gyroscope()[i];
  for(int isaac=0;isaac<kDof;++isaac){const int hw=kMujocoToIsaac[isaac];out[3+isaac]=state.motor_state()[hw].q()-kDefaultAngles[hw];out[32+isaac]=state.motor_state()[hw].dq();out[61+isaac]=last_action[isaac];}
  const auto g=gravity({state.imu_state().quaternion()[0],state.imu_state().quaternion()[1],state.imu_state().quaternion()[2],state.imu_state().quaternion()[3]});
  for(int i=0;i<3;++i) out[90+i]=g[i];
  return out;
}

void build_observations(const Motion& motion, int frame, bool playing, const std::deque<std::array<float,93>>& history,
                        Quat base, Quat heading_base, std::vector<float>& enc, std::vector<float>& dec) {
  enc.assign(kEncoderInput,0.f); enc[0]=0.f; const int f=playing?frame:0;
  for(int n=0;n<10;++n){const int idx=std::min(f+n,motion.frames()-1);for(int j=0;j<kDof;++j){enc[4+n*kDof+j]=motion.pos[idx][j];enc[294+n*kDof+j]=playing?motion.vel[idx][j]:0.f;}
    const Quat initial=motion.root[0]; const Quat delta=qmul(heading(heading_base),qconj(heading(initial))); const Quat rel=qmul(qconj(base),qmul(delta,motion.root[idx])); auto r=rot6(rel);for(int j=0;j<6;++j)enc[584+n*6+j]=r[j];}
  dec.assign(kDecoderInput,0.f); int n=0; for(const auto& s:history){for(int j=0;j<3;++j)dec[64+n*3+j]=s[j];for(int j=0;j<kDof;++j){dec[94+n*kDof+j]=s[3+j];dec[384+n*kDof+j]=s[32+j];dec[674+n*kDof+j]=s[61+j];}for(int j=0;j<3;++j)dec[964+n*3+j]=s[90+j];++n;}
}

Command make_command(const std::array<float,kDof>& action) { Command c; for(int hw=0;hw<kDof;++hw){const int isaac=kIsaacToMujoco[hw];c.q[hw]=kDefaultAngles[hw]+kActionScale[hw]*action[isaac];c.kp[hw]=kKp[hw];c.kd[hw]=kKd[hw];} return c; }
Command damping() { Command c; for(int i=0;i<kDof;++i)c.kd[i]=8.f; return c; }

class Controller {
 public:
  Controller(std::string encoder,std::string decoder,std::string motion,int domain,std::string iface,bool auto_start,bool auto_play)
      : encoder_("encoder",std::move(encoder),kEncoderInput,kToken),decoder_("decoder",std::move(decoder),kDecoderInput,kAction),motion_(std::move(motion)),domain_(domain),iface_(std::move(iface)),auto_start_(auto_start),auto_play_(auto_play) {}
  void run() {
    encoder_.open(); decoder_.open();
    ChannelFactory::Instance()->Init(domain_,iface_);
    publisher_=std::make_unique<ChannelPublisher<LowCmd>>("rt/lowcmd");publisher_->InitChannel();
    subscriber_=std::make_unique<ChannelSubscriber<LowState>>("rt/lowstate");subscriber_->InitChannel([this](const void* p){buffer_.update(*static_cast<const LowState*>(p));},1);
    std::atomic_store(&command_,std::make_shared<Command>(damping()));
    std::thread writer([this]{write_loop();});
    init_loop();
    control_loop();
    stop_=true; writer.join();
  }
 private:
  void write_loop(){const auto period=std::chrono::microseconds(2000);auto next=Clock::now();while(!stop_){auto snap=buffer_.latest();auto cmd=std::atomic_load(&command_);if(cmd){LowCmd out;out.mode_machine()=snap?snap->state.mode_machine():mode_machine_;out.mode_pr()=0;for(int i=0;i<kDof;++i){out.motor_cmd()[i].mode()=1;out.motor_cmd()[i].q()=cmd->q[i];out.motor_cmd()[i].dq()=cmd->dq[i];out.motor_cmd()[i].tau()=cmd->tau[i];out.motor_cmd()[i].kp()=cmd->kp[i];out.motor_cmd()[i].kd()=cmd->kd[i];}out.crc()=crc32(reinterpret_cast<uint32_t*>(&out),(sizeof(LowCmd)>>2)-1);publisher_->Write(out);}next+=period;std::this_thread::sleep_until(next);}}
  void init_loop(){std::cout<<"state=INIT\n";auto begin=Clock::now();while(true){auto s=buffer_.latest();if(!s){std::this_thread::sleep_for(std::chrono::milliseconds(20));continue;}mode_machine_=s->state.mode_machine();double ratio=std::min(1.0,std::chrono::duration<double>(Clock::now()-begin).count()/3.0);auto c=std::make_shared<Command>();for(int i=0;i<kDof;++i){const float q=s->state.motor_state()[i].q();c->q[i]=q*(1-ratio)+kDefaultAngles[i]*ratio;c->kp[i]=kKp[i];c->kd[i]=kKd[i];}std::atomic_store(&command_,c);if(ratio>=1.0)break;std::this_thread::sleep_for(std::chrono::milliseconds(20));}std::cout<<"state=WAIT_FOR_CONTROL\n";}
  void control_loop(){std::deque<std::array<float,93>> history(10);std::array<float,kDof> last{};int frame=0;bool playing=auto_play_;bool started=auto_start_;int tick=0;Quat heading_base{};bool heading_valid=false;auto next=Clock::now();while(!stop_){auto s=buffer_.latest();if(!s||Clock::now()-s->received>std::chrono::milliseconds(500)){std::cerr<<"state=ERROR lowstate stale\n";std::atomic_store(&command_,std::make_shared<Command>(damping()));return;}if(!started){if(auto_start_)started=true;else{std::this_thread::sleep_for(std::chrono::milliseconds(20));continue;}}auto sf=state_frame(s->state,last);history.push_back(sf);if(history.size()>10)history.pop_front();Quat base{ s->state.imu_state().quaternion()[0],s->state.imu_state().quaternion()[1],s->state.imu_state().quaternion()[2],s->state.imu_state().quaternion()[3]};if(!heading_valid){heading_base=base;heading_valid=true;}std::vector<float> enc,dec;build_observations(motion_,frame,playing,history,base,heading_base,enc,dec);auto token=encoder_.run(enc);std::copy(token.begin(),token.end(),dec.begin());auto raw=decoder_.run(dec);if(raw.size()!=kAction||!std::all_of(raw.begin(),raw.end(),[](float x){return std::isfinite(x);})){std::cerr<<"state=ERROR invalid action\n";std::atomic_store(&command_,std::make_shared<Command>(damping()));return;}std::copy(raw.begin(),raw.end(),last.begin());std::array<float,kDof> action{};std::copy(raw.begin(),raw.end(),action.begin());auto policy_command=std::make_shared<Command>(make_command(action));std::atomic_store(&command_,policy_command);if(++tick==1||tick%50==0){float max_action=0.f;for(float x:raw)max_action=std::max(max_action,std::abs(x));std::cout<<"state=CONTROL tick="<<tick<<" frame="<<frame<<" playing="<<playing<<" action_abs_max="<<max_action<<" q0="<<policy_command->q[0]<<"\n";}if(playing&&++frame>=motion_.frames()){playing=false;frame=0;heading_valid=false;}next+=std::chrono::milliseconds(20);std::this_thread::sleep_until(next);}}
  RknnModel encoder_,decoder_;Motion motion_;int domain_;std::string iface_;bool auto_start_,auto_play_;StateBuffer buffer_;std::unique_ptr<ChannelPublisher<LowCmd>> publisher_;std::unique_ptr<ChannelSubscriber<LowState>> subscriber_;std::shared_ptr<Command> command_;std::atomic<bool> stop_{false};uint8_t mode_machine_=0;
};

int main(int argc,char**argv){try{std::string enc="sonic_encoder_int8.rknn",dec="sonic_decoder_int8.rknn",motion="reference/dance_in_da_party_001__A464_M",iface="eth0";int domain=0;bool start=false,play=false;for(int i=1;i<argc;++i){std::string a=argv[i];auto next=[&]{if(i+1>=argc)throw std::runtime_error("missing argument");return std::string(argv[++i]);};if(a=="--encoder")enc=next();else if(a=="--decoder")dec=next();else if(a=="--motion")motion=next();else if(a=="--domain")domain=std::stoi(next());else if(a=="--interface")iface=next();else if(a=="--auto-start")start=true;else if(a=="--auto-play")play=true;else if(a=="--help"){std::cout<<"usage: sonic_rk3576_controller [--encoder p] [--decoder p] [--motion dir] [--domain n] [--interface nic] [--auto-start] [--auto-play]\n";return 0;}else throw std::runtime_error("unknown arg "+a);}Controller(enc,dec,motion,domain,iface,start,play).run();return 0;}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
