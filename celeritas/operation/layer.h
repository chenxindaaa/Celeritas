class BaseLayer {
   public:
    explicit BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                       std::string layer_name = "");

    base::DataType data_type() const;

    LayerType layer_type() const;

    virtual base::Status init() = 0;

    virtual base::Status forward() = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& output1) = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& input4,
                                 const tensor::Tensor& output1) = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& input4,
                                 const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;

    virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

    virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

    virtual size_t input_size() const = 0;

    virtual size_t output_size() const = 0;

    virtual base::Status check() const = 0;

    virtual tensor::Tensor& get_input(int32_t idx) = 0;

    virtual tensor::Tensor& get_output(int32_t idx) = 0;

    virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

    virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

    virtual base::Status set_weight(int32_t idx, const tensor::Tensor& weight);

    virtual base::Status set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr,
                                    base::DeviceType device_type = base::DeviceType::kDeviceUnknown);

    const std::string& get_layer_name() const;

    void set_layer_name(const std::string& layer_name);

    base::DeviceType device_type() const;

    void set_device_type(base::DeviceType device_type);

   protected:
    std::string layer_name_;
    LayerType layer_type_ = LayerType::kLayerUnknown;
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
};