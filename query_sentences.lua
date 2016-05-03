stringx = require('pl.stringx')
require 'io'
require 'cunn'
require 'nn'
require 'nngraph'
require('./base')
data = require('data')

model = torch.load('model.t7')

function reset_state()
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * 2 do
       model.start_s[d]:zero()
    end
  end
end

function inData(token)
  if data.inverse_map(token) == nil then return false end
  return true 
end

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  for i = 2,#line do
    if not inData(line[i]) then error({code="vocab", word = line[i]}) end
  end
  return line
end

function get_index(line)
  local idx_seq = line:clone()
  for i = 2, #line do
    if data.inverse_map(line[i]) == nil then idx_seq[i] = data.inverse_map("<unk>")
    else idx_seq[i] = data.inverse_map(line[i])
    end
  end
  return idx_seq
end

function make_prediction(seq)
  local known_len = #seq-1
  local pred_len = tonumber(seq[1])
  reset_state()
  g_disable_dropout(model.rnns)
  g_replace_table(model.s[0], model.start_s)
  for i = 2, known_len do
    local x = torch.CudaTensor(batch_size):fill(seq[i])
    local y = torch.CudaTensor(batch_size):fill(seq[i+1])
    _, model.s[1], _ = unpack(model.rnns[1]:forward({x,y,model.s[0]}))
    g_replace_table(model.s[0], model.s[1])
  end
  local pred_seq = torch.Tensor(pred_len)
  pred_seq:cuda()
  local x = torch.CudaTensor(batch_size):fill(seq[known_len+1])
  local y = torch.CudaTensor(batch_size):fill(seq[known_len+1])
  for i = 1, pred_len do
    _, model.s[1], pred = unpack(model.rnns[1]:forward({x,y,model.s[0]}))
    if criterion == "max" then
      _, pred_tensor = torch.max(pred, 2)
      x = torch.CudaTensor(batch_size):fill(pred_tensor[1][1])
      pred_seq[i] = pred_tensor[1][1]
    elseif criterion == "multinomial" then
      local pred_log = pred[1]
      local pred_exp = torch.exp(pred_log)
      local pred_ind = torch.multinomial(pred_exp, 1)[1]
      x = torch.CudaTensor(batch_size):fill(pred_ind)
      pred_seq[i] = pred_ind
    end
    g_replace_table(model.s[0], model.s[1])
  end
  return pred_seq
end

function print_prediction(input_seq, pred_seq)
  local output = input_seq[2]
  for i = 3, #input_seq do
    output = output .. " " .. input_seq[i]
  end
  for i = 1, pred_seq:size(1) do
    local ind = pred_seq[i]
    local word = data.reverse_map[ind]
    output = output .. " " .. word
  end
  print(output)
end

num_layers=2
batch_size=20
criterion="multinomial"
model_type="lstm"


while true do
  print("Query: len word1 word2 etc")
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "vocab" then
      print("Word not in vocabulary, only 'foo' is in vocabulary: ", line.word)
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
	local x=get_index(line)
	local s=make_prediction(x)
	print_prediction(line,s)
  end
end
