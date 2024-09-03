# import tflite
import tflite.Model
# import tflite.SubGraph
import flatbuffers
from operator import itemgetter

from io import StringIO
from mako.template import Template
import os

def plan_memory(model):
	tensor_expected_buffer_size = {}

	subgraph = model.Subgraphs(0)
	for i in range(subgraph.TensorsLength()):
		tensor = subgraph.Tensors(i)
		import tflite.TensorType
		tensor_type = tensor.Type()
		if tensor_type == tflite.TensorType.TensorType.INT8:
			tensor_data_uint_size = 1
		elif tensor_type == tflite.TensorType.TensorType.UINT8:
			tensor_data_uint_size = 1
		elif tensor_type == tflite.TensorType.TensorType.INT32:
			tensor_data_uint_size = 4
		elif tensor_type == tflite.TensorType.TensorType.UINT32:
			tensor_data_uint_size = 4
		else:
			tensor_data_uint_size = 0
			print("Cannot handle the data type of tensor {0}!".format(i))

		tensor_elements_count = 1
		for j in range(tensor.ShapeLength()):
			tensor_elements_count  = tensor_elements_count * tensor.Shape(j)
		tensor_expected_buffer_size[tensor.Buffer()] = tensor_elements_count * tensor_data_uint_size

	#key: buffer index, value: mem type (RW or RO)
	buffer_mem_type = {}
	# key: buffer index, value: required size
	buffer_to_alloc = {}
	for i in range(model.BuffersLength()):
		buffer = model.Buffers(i)
		if i in tensor_expected_buffer_size:
			if tensor_expected_buffer_size[i] == buffer.DataLength():
				buffer_mem_type[i] = 'RO'
			else:
				buffer_mem_type[i] = 'RW'
				buffer_to_alloc[i] = tensor_expected_buffer_size[i]

	return buffer_mem_type, buffer_to_alloc

def opcode_to_name(code):
	import tflite.BuiltinOperator
	for name, value in vars(tflite.BuiltinOperator.BuiltinOperator).items():
		if value == code:
			return name
	return None

def write_opcode(buf, opcode_list):
	# write array to store kernels ptr
	buf.write(
		f"TFLMRegistration reg[{len(opcode_list)}];\n"
	)
	# write kernel binding function
	buf.write('void op_init(void){')
	for i in range(len(opcode_list)):
		opcode_name = opcode_to_name(opcode_list[i])
		buf.write(f'reg[{i}] = tflite::Register_{opcode_name}();\n')
	buf.write('}')
	return

def gen_kernel(model):
	kernel_str = StringIO()
	opcode_list = []
	for i in range(model.OperatorCodesLength()):
		opcode_idx = model.OperatorCodes(i).BuiltinCode()
		opcode_list.append(opcode_idx)
	write_opcode(kernel_str, opcode_list)
	kernel_str.seek(0)
	return kernel_str.read()

def gen_buffer(model, buffer_mem_type, buffer_to_alloc):
	buffer_str = StringIO()
	for i in range(model.BuffersLength()):
		if i in buffer_mem_type:
			if buffer_mem_type[i] == 'RO':
				buffer_str.write(f'const uint8_t buffer_{i}[] = {{')
				for j in range(model.Buffers(i).DataLength()):
					buffer_str.write(f'{hex(model.Buffers(i).Data(j))},')
				buffer_str.write('};')
			else:
				buffer_str.write(f'uint8_t buffer_{i}[{buffer_to_alloc[i]}];')
	buffer_str.seek(0)
	return buffer_str.read()

def gen_tensor(model, buffer_mem_type):
	tensor_str = StringIO()
	for i in range(model.Subgraphs(0).TensorsLength()):
		tensor = model.Subgraphs(0).Tensors(i)
		tensor_dims = []
		for j in range(tensor.ShapeLength()):
			tensor_dims.append(tensor.Shape(j))
		tensor_str.write(f'const TfArray<{len(tensor_dims)}, int> dim_{i} = {{{len(tensor_dims)},  {{{str(tensor_dims)[1:-1]}}}  }};')

	tensor_str.write(f'TfLiteTensor tflTensors[{model.Subgraphs(0).TensorsLength()}] ={{')
	for i in range(model.Subgraphs(0).TensorsLength()):
		tensor = model.Subgraphs(0).Tensors(i)
		tensor_str.write('{')
		if tensor.Quantization():
			tensor_str.write('.quantization = {.type = kTfLiteAffineQuantization, },')
			tensor_str.write(f'.params = {{.scale = {tensor.Quantization().Scale(0)}, .zero_point = {tensor.Quantization().ZeroPoint(0)}}},')
		else:
			tensor_str.write('.quantization = {.type = kTfLiteNoQuantization, },')

		tensor_str.write(f'.data = {{.data = (void*)buffer_{tensor.Buffer()}}},')
		tensor_str.write(f'.dims = (TfLiteIntArray*)&dim_{i},')

		tensor_elements_count = 1
		for j in range(tensor.ShapeLength()):
			tensor_elements_count  = tensor_elements_count * tensor.Shape(j)

		tensor_type = tensor.Type()
		import tflite.TensorType
		if tensor_type == tflite.TensorType.TensorType.INT8:
			tensor_type_str = 'kTfLiteInt8'
			tensor_data_uint_size = 1
		elif tensor_type == tflite.TensorType.TensorType.UINT8:
			tensor_type_str = 'kTfLiteUInt8'
			tensor_data_uint_size = 1
		elif tensor_type == tflite.TensorType.TensorType.INT32:
			tensor_type_str = 'kTfLiteInt32'
			tensor_data_uint_size = 4
		elif tensor_type == tflite.TensorType.TensorType.UINT32:
			tensor_type_str = 'kTfLiteUInt32'
			tensor_data_uint_size = 4
		else:
			tensor_type_str = 'UNKNOWN'
			tensor_data_uint_size = 0
			print("Cannot handle!")
		tensor_str.write(f'.bytes = {tensor_elements_count * tensor_data_uint_size},')
		tensor_str.write(f'.type = {tensor_type_str},')
		if buffer_mem_type[tensor.Buffer()] == 'RW':
			alloc_type_str = 'kTfLiteArenaRw'
		else:
			alloc_type_str = 'kTfLiteMmapRo'
		tensor_str.write(f'.allocation_type = {alloc_type_str},')
		tensor_str.write(f'.is_variable = 0,')
		tensor_str.write('},')


	tensor_str.write('};')

	tensor_str.write(f'TfLiteEvalTensor tflEvalTensors[{model.Subgraphs(0).TensorsLength()}] ={{')
	for i in range(model.Subgraphs(0).TensorsLength()):
		tensor = model.Subgraphs(0).Tensors(i)
		tensor_str.write('{')
		tensor_str.write(f'.data = {{.data = (void*)buffer_{tensor.Buffer()}}},')
		tensor_str.write(f'.dims = (TfLiteIntArray*)&dim_{i},')

		tensor_type = tensor.Type()
		if tensor_type == tflite.TensorType.TensorType.INT8:
			tensor_type_str = 'kTfLiteInt8'
		elif tensor_type == tflite.TensorType.TensorType.UINT8:
			tensor_type_str = 'kTfLiteUInt8'
		elif tensor_type == tflite.TensorType.TensorType.INT32:
			tensor_type_str = 'kTfLiteInt32'
		elif tensor_type == tflite.TensorType.TensorType.UINT32:
			tensor_type_str = 'kTfLiteUInt32'
		else:
			tensor_type_str = 'UNKNOWN'
			print("Cannot handle!")
		tensor_str.write(f'.type = {tensor_type_str},')

		tensor_str.write('},')

	tensor_str.write('};')

	tensor_str.seek(0)
	return tensor_str.read()

def gen_node(model):
	node_str = StringIO()
	opcode_list = []
	for i in range(model.Subgraphs(0).OperatorsLength()):
		opcode_list.append(model.Subgraphs(0).Operators(i).OpcodeIndex())
	node_str.write(f'int op_code_index[] = {{    {str(opcode_list)[1:-1]}     }};')

	for i in range(model.Subgraphs(0).OperatorsLength()):
		node_input = []
		node_output = []
		for j in range(model.Subgraphs(0).Operators(i).InputsLength()):
			node_input.append(model.Subgraphs(0).Operators(i).Inputs(j))
		for j in range(model.Subgraphs(0).Operators(i).OutputsLength()):
			node_output.append(model.Subgraphs(0).Operators(i).Outputs(j))
		node_str.write(f'const TfArray<{len(node_input)}, int> input_{i} = {{{len(node_input)},  {{{str(node_input)[1:-1]}}}  }};')
		node_str.write(f'const TfArray<{len(node_output)}, int> output_{i} = {{{len(node_output)},  {{{str(node_output)[1:-1]}}}  }};')

	import tflite.BuiltinOptions
	import tflite.FullyConnectedOptions
	import tflite.ActivationFunctionType
	opdata_presence = []
	for i in range(model.Subgraphs(0).OperatorsLength()):
		if model.Subgraphs(0).Operators(i).BuiltinOptionsType() == tflite.BuiltinOptions.BuiltinOptions.FullyConnectedOptions:
			fully_connected_param = tflite.FullyConnectedOptions.FullyConnectedOptions()
			tflite.FullyConnectedOptions.FullyConnectedOptions.Init(fully_connected_param, model.Subgraphs(0).Operators(i).BuiltinOptions().Bytes, model.Subgraphs(0).Operators(i).BuiltinOptions().Pos)
			node_str.write(f'const TfLiteFullyConnectedParams opdata_{i} = {{')
			if fully_connected_param.FusedActivationFunction() == tflite.ActivationFunctionType.ActivationFunctionType.NONE:
				node_str.write('kTfLiteActNone,')
			elif fully_connected_param.FusedActivationFunction() == tflite.ActivationFunctionType.ActivationFunctionType.RELU:
				node_str.write('kTfLiteActRelu,')
			node_str.write('kTfLiteFullyConnectedWeightsFormatDefault,')
			node_str.write('false,')
			node_str.write('false,')
			node_str.write('};')
			opdata_presence.append(i)
		else:
			print("Unsupported Operator built-in options!")



	node_str.write(f'TfLiteNode my_node[] = {{')
	for i in range(model.Subgraphs(0).OperatorsLength()):
		node_str.write('{')
		node_str.write(f'.inputs = (TfLiteIntArray*)&input_{i},')
		node_str.write(f'.outputs = (TfLiteIntArray*)&output_{i},')
		if i in opdata_presence:
			node_str.write(f'.builtin_data = (void*)&opdata_{i},')
		node_str.write('.custom_initial_data = nullptr,')
		node_str.write('.custom_initial_data_size = 0,')
		node_str.write('},')
	node_str.write('};')

	node_str.seek(0)
	return node_str.read()


def main():
	buf = open('hello_world_int8.tflite', 'rb').read()
	buf = bytearray(buf)
	model = tflite.Model.Model.GetRootAsModel(buf, 0)
	buffer_mem_type, buffer_to_alloc = plan_memory(model)
	kernel_str = gen_kernel(model)
	buffer_str = gen_buffer(model, buffer_mem_type, buffer_to_alloc)
	tensor_str = gen_tensor(model, buffer_mem_type)
	node_str = gen_node(model)

	mytemplate = Template(filename='model.cc.mako')

	f = open('model.cc', 'w')
	f.write(mytemplate.render(kernel_region=kernel_str,\
				buffer_region=buffer_str,\
				tensor_region=tensor_str,\
				node_region=node_str,\
				node_count=model.Subgraphs(0).OperatorsLength()))
	f.close()
	os.system('clang-format -i model.cc')


if __name__ == "__main__":
  main()