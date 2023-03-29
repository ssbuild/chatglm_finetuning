void compress_int4_weight(void *weight, void *out, int n, int m)
{
    for(int i=0;i<n*m;i++)
    {
        (*(unsigned char*)(out)) = ((*(unsigned char*)(weight)) << 4);
        weight += sizeof(char);
        (*(unsigned char*)(out)) |= ((*(unsigned char*)(weight)) & 15);
        weight += sizeof(char);
        out += sizeof(char);
    }
}

void extract_int8_weight_to_float(void *weight, void *scale_list, void *out, int n, int m)
{
	for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            (*(float*)(out + sizeof(float) * (i * m + j))) = (*(float*)(scale_list + sizeof(float) * i)) * (*(char*)(weight + sizeof(char) * (i * m + j)));
}

void extract_int4_weight_to_float(void *weight, void *scale_list, void *out, int n, int m)
{
	for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
        {
            (*(float*)(out)) = (*(float*)(scale_list)) * ((*(char*)(weight)) >> 4);
            out += sizeof(float);
            (*(float*)(out)) = (*(float*)(scale_list)) * (((char)((*(unsigned char*)(weight)) << 4))>> 4);
            out += sizeof(float);
            weight += sizeof(char);
        }
        scale_list += sizeof(float);
    }
}
